# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``dual_stream_wan`` 族：video + action 两条等深流 + 一张联合注意力。

族级装配以 ``docs/design/mot_spec_model_matrix.zh-CN.md`` §4.2 为准，
``vpp2`` provider 以 §4.3 / §4.4 为准。VPP2 的代码级事实以
``projects/video-prediction-policy-2/src/vpp2/models/wan21_14b/`` 下的
``vpp2.py`` / ``mot.py`` 为准，逐处标注行号。

族里三个模型：``vpp2`` / ``fastwam`` / ``internw0_delta``。
``_dual_stream_wan_streams`` **不注册** —— 它只是三者的公共前半段，
YAML 与启动参数里都看不见它（spec_config §3.8）。

**三个 impl 工厂**（``wan_video_stream`` / ``action_stream`` / ``flow_head``）
是 ``Spec.impl`` 的落点。它们现在抛 ``NotImplementedError``：现状的
``WanVideoDiT`` / ``ActionDiT`` 是**整模型**而不是一条流，拆成
``StreamModule`` 的四段属 Lane B。``Spec.impl`` 要求可调用对象而不是字符串，
所以这三个名字必须在 import 期就存在。
"""

from __future__ import annotations

from typing import Any

from ...diffusion.loss import LossSpec, WeightRamp
from ...diffusion.schedule import RectifiedFlow
from ...diffusion.spec import DiffusionSpec
from ...diffusion.stepping import EulerSampler, UniformSteps
from ...diffusion.timesteps import Constant, TimestepSampler, UniformShift
from ...geometry import GeometrySpec
from ...heads.hidden_ref import HiddenStateRef
from ...heads.spec import HeadSpec
from ...streams.conditions import ConditionSpec, InjectionKind
from ...streams.engine import AttentionGroup
from ...streams.spec import StreamSpec
from ...visibility.combine import Visibility
from ...visibility.presets import get_preset
from ...visibility.rules import frame_causal, prefix_only, same_stream, stream_edge
from ..core import Spec, register_provider
from ..world import WorldModelSpec

__all__ = ["wan_video_stream", "action_stream", "flow_head"]


# ---- 三个 impl 工厂。build(impl, spec=..., geometry=...) 调它们 ----


def wan_video_stream(*, spec: StreamSpec, geometry: GeometrySpec) -> Any:
    """``video`` 流的 block。包的是 ``model/fastwam/wan/dit.py:512``
    的 ``WanVideoDiT``（VPP2 侧同名类）。

    它现在是一个整模型：入口段与出口段写在 ``pre_dit`` / ``post_dit``
    （VPP2 ``vpp2.py:1325`` / ``:1362``），逐层那段在 ``MoT.forward``
    （``mot.py:811-957``）。拆成 ``StreamModule`` 的
    prepare / pre_attn / post_attn / finalize 四段属 Lane B。
    """
    raise NotImplementedError(
        f"wan_video_stream（流 {spec.name!r}）属 Lane B（L2 流模块，S2 交付）；"
        "判据见 docs/design/mot_l2_streams_engine.detailed.zh-CN.md"
    )


def action_stream(*, spec: StreamSpec, geometry: GeometrySpec) -> Any:
    """``action`` 流的 block。包的是 ``model/fastwam/action/dit.py:57``
    的 ``ActionDiT``。

    这条流是 ``residual_width != attn_width`` 的直接证据来源：
    VPP2 残差 512、注意力 40 × 128 = 5120。
    """
    raise NotImplementedError(
        f"action_stream（流 {spec.name!r}）属 Lane B（L2 流模块，S2 交付）"
    )


def flow_head(*, spec: HeadSpec, geometry: GeometrySpec) -> Any:
    """flow matching 的回归头。输出宽度取 ``spec.output_width`` ——
    样例里的 ``Spec(FlowHead, {"out_dim": 16})`` 那个参数在本实现里
    **不从 params 来**：它已经是 ``HeadSpec.output_width``，
    两处各写一遍就会配出不一致的值（V8 校验的正是这一致性）。
    """
    raise NotImplementedError(
        f"flow_head（头 {spec.name!r}，output_width={spec.output_width}）"
        "属 Lane C（L4 输出头，S3 交付）"
    )


# ---- 族级共享：两条流的装配（§4.2）----


def _dual_stream_wan_streams(
    video_impl: Spec,
    action_impl: Spec,
    video_residual_width: int = 3072,
    video_num_heads: int = 24,
    video_head_dim: int = 128,
    video_num_layers: int = 30,
    video_ffn_width: int = 14336,
    action_residual_width: int = 1024,
    action_num_heads: int = 24,
    action_head_dim: int = 128,
    action_num_layers: int = 30,
    action_ffn_width: int = 4096,
    cross_kv_slots: tuple[str, ...] = ("text",),
) -> tuple[StreamSpec, StreamSpec]:
    """族内三个 provider 共享的两条流。拓扑写死在这里，数值留形参。

    默认值取基线 fastwam 的实测取值
    （``model/fastwam/modeling_configuration_fastwam.py:75-96``）——
    注意 action 流残差 1024 而注意力 24 × 128 = 3072，两者本来就不等。

    ``impl`` 两个都必填：族里三个 provider 的 block 类各不相同，
    族级给不出默认值。

    两条流的 ``num_layers`` 必须相等 —— 这是逐层联合注意力的前提。
    §4.2 正文说它由 V3 保证，那是错的（V3 管 ``head_dim`` 相等）；
    **V2 也只挡住一半**。实测（``resolve_layer_index_map``，
    ``streams/spec.py:263-290``）：组的联合层数取 ``video.num_layers`` 时，
      - action 流**多于**联合层数 ⇒ V2 抛错；
      - action 流**少于**联合层数 ⇒ V2 **通过** —— 在 ONE_TO_ONE 下
        ``participates_in_joint`` 判的是 ``layer_id < self.num_layers``
        （``:253``），所以一条浅流会被解释成「只参与前 N 个联合层」，
        那是 STRIDED / AT_LAYERS 要表达的合法形态，V2 没有理由拒绝它。
    族里三个成员都不是那种形态，所以这条前提只能由**族自己**声明。
    下面那个显式相等校验就是它，不要删掉改回去依赖 V2。

    ``cross_kv_slots`` 两条流同值：族里三个成员的 block 都是 WAN DiT 形态，
    逐层带一个 text cross-attn（VPP2 侧 ``wan_video_dit.py`` 的 ``context``
    与 ``action_dit.py:405`` 的 ``text_embedding``）。它进这里而不是进
    provider，是因为「有几个 cross-attn 槽」属族级拓扑。
    """
    if video_num_layers != action_num_layers:
        raise ValueError(
            "dual_stream_wan 族要求两条流等深（逐层联合注意力的前提），"
            f"得到 video_num_layers={video_num_layers}、"
            f"action_num_layers={action_num_layers}。"
            "V2 只挡住 action 比联合层数多的那一侧，少的那一侧它会当成"
            "「只参与前几层的浅流」而放过 —— 所以这条必须在族里查"
        )
    return (
        StreamSpec(
            name="video",
            impl=video_impl,
            residual_width=video_residual_width,
            num_heads=video_num_heads,
            head_dim=video_head_dim,
            num_layers=video_num_layers,
            ffn_width=video_ffn_width,
            cross_kv_slots=cross_kv_slots,
            writable=True,
        ),
        StreamSpec(
            name="action",
            impl=action_impl,
            residual_width=action_residual_width,
            num_heads=action_num_heads,
            head_dim=action_head_dim,
            num_layers=action_num_layers,
            ffn_width=action_ffn_width,
            cross_kv_slots=cross_kv_slots,
            writable=True,
        ),
    )


_STREAM_SCALAR_KEYS: frozenset[str] = frozenset({
    "video_residual_width", "video_num_heads", "video_head_dim",
    "video_num_layers", "video_ffn_width",
    "action_residual_width", "action_num_heads", "action_head_dim",
    "action_num_layers", "action_ffn_width",
})
"""``_dual_stream_wan_streams`` 的十个数值形参名。与那个签名**手工对齐** ——
用 ``inspect.signature`` 反推会把 ``video_impl`` / ``cross_kv_slots`` 一起扫进来，
而那两个不是 [YAML] 可覆写的东西。"""


def _resolve_scalars(model_type: str, defaults: dict[str, int],
                     overrides: dict[str, Any]) -> dict[str, int]:
    """施加流标量覆写，未被消费的键抛错。

    严格而不是静默忽略，是 spec_config §3.8 的第三条：静默会把一次拼错的键名
    变成一次「配了但没生效」，而那只能靠读 provider 源码才能发现。

    ``defaults`` 只装**与族默认不同**的那几个值；空表合法（fastwam 就是空的）。
    合法键集取 ``_STREAM_SCALAR_KEYS`` 而不是 ``set(defaults)`` ——
    否则 provider 侧没覆写的那几个标量就连 YAML 也覆写不了，
    而「能不能被 YAML 覆写」是族级事实，不该随 provider 的默认表变。
    ``defaults`` 自己的键集也一并校验：写漏/写错一个键名的症状，
    否则是「这个值永远生效不了」。
    """
    for label, keys in (("默认表含非法键", set(defaults)),
                        ("收到了它不消费的键", set(overrides))):
        stray = sorted(keys - _STREAM_SCALAR_KEYS)
        if stray:
            raise ValueError(
                f"provider({model_type!r}) {label} {stray}；"
                f"合法的流标量为 {sorted(_STREAM_SCALAR_KEYS)}"
                "（spec_config §3.8）"
            )
    resolved = dict(defaults)
    resolved.update(overrides)
    return resolved


def _reject_config(model_type: str, config: Any) -> None:
    """``build_world_model`` 的调用约定是 ``provider(config)``，但 YAML 覆写的
    施加是它第③步、属配置层，尚未落地 ⇒ 对非 None 的 config 抛错而不是静默忽略。
    """
    if config is not None:
        raise NotImplementedError(
            f"provider({model_type!r}) 收到了配置对象，但 YAML 覆写的施加是 "
            "build_world_model 第③步（属配置层，判据见 "
            "mot_spec_config.zh-CN.md §3.8 的 R1–R5）。"
            f"在那之前请直接调 provider 取默认拓扑"
        )


# ---- provider：vpp2（§4.3 / §4.4）----

_VPP2_STREAM_SCALARS: dict[str, int] = {
    # 一手出处：``configs/libero/train_action.yaml`` 的 video_dit_config
    # （:169-190）与 action_dit_config（:191-200）。与 model_matrix §3 的
    # VPP2 行一致：video 5120/5120、action 512/5120
    "video_residual_width": 5120,
    "video_num_heads": 40,
    "video_head_dim": 128,
    "video_num_layers": 40,
    "video_ffn_width": 13824,
    "action_residual_width": 512,
    "action_num_heads": 40,
    "action_head_dim": 128,
    "action_num_layers": 40,
    "action_ffn_width": 2048,
}
"""VPP2 的流标量。它们是 [YAML] 的，默认值写在 provider 里（§4.1 的 15 行
YAML 逐字段覆写这张表）。族默认值是 fastwam 的，VPP2 与它不同族值。"""


def _vpp2_visibility(visible_video_frames: int) -> dict[str, Visibility]:
    """VPP2 的 L1。键是**流名**，值是该流 query 侧的规则。

    一手事实（``vpp2.py:879-906``，``[S, S]`` bool 共三次写入）：
      - ``:889`` video → video；
      - ``:894`` action → action 全 True；
      - ``:905`` action → 视频前 N 帧
        （``visible_video_tokens = tokens_per_frame × visible_frames``）。
    **没有 video → action 这条边** ⇒ ``video`` 侧是 ``same_stream()``，
    这也正是推理期可以只跑动作头的结构原因。

    三处与 §4.3 样例的差异，都要在等价性测试前裁定（本函数按样例写，
    差异在此记账，不静默取一侧）：
      1. 样例有 ``chunk_causal()``，现状 ``:894`` 是全 True —— 多这条规则
         会遮掉 action 自己的未来块；
      2. 样例注释说「最近 N 帧」，现状 ``:905`` 取的是**前** N 帧（前缀）。
         滑动窗口对应 ``frame_causal(N)``，前缀对应 ``prefix_only(n)``；
      3. 现状的 clean 条件是两个返回 ``[B]`` bool 的方法
         （``:908-921`` / ``:923-936``），按 INV-1 不得进 ``structural``；
         ``noise_gate`` 把它升级成逐 token 标签上的规则，是 12/04 闸门的对象。
    """
    noise_gated_history = get_preset("noise_gated_history")
    return {
        "video": same_stream(),
        "action": same_stream() | (
            stream_edge("video", "action")
            & noise_gated_history(window=visible_video_frames)
        ),
    }


@register_provider("vpp2")
def _vpp2(config: Any = None, *,
          visible_video_frames: int = 3,
          action_shift: float = 5.0,
          inference_steps: int = 10,
          video_latent_channels: int = 16,
          action_width: int = 7,
          **stream_scalars: Any) -> WorldModelSpec:
    """``--model-name vpp2`` 真正落到的那个函数。

    **与 §4.3 / §4.4 代码样例的签名差异**（样例是示意，签名以
    ``mot_layer_interfaces.zh-CN.md`` 与已落地的 dataclass 为准）。
    九处，逐条记账而不是静默取一侧：

    1. ``GeometrySpec(image_size=, temporal_ratio=)`` →
       ``frame_height=`` / ``frame_width=`` / ``action_video_freq_ratio=``。
       interfaces §3.1 的字段表；高宽分开是因为存在非正方形输入。
    2. ``WorldModelSpec(streams=, routes=)`` →
       ``groups=(AttentionGroup(members=…, num_layers=…, routes=…),)``。
       「哪些流在同一张联合注意力里」必须在类型里可见（EWAM 的 VL 流不在）。
    3. 逐流两个 ``DiffusionSpec`` → 一个 ``DiffusionSpec``，``timesteps``
       按流名索引、``losses`` 是列表。损失条数逐模型不同（2/3/5 条），
       列表才装得下。
    4. ``DiffusionSpec(timestep=, loss=, sample_shape=)`` → ``timesteps=`` /
       ``losses=``；「采多少个 t」由 ``TimestepSampler.broadcast.axes``
       决定，不另设 ``sample_shape``。
    5. ``LossSpec(term=, reduce=, ignore_pad=)`` → ``LossSpec(name=, head=)``。
       ``reduce`` / ``ignore_pad`` 归 ``broadcast`` 与 ``pad`` 标签（I2），
       不在 loss 上再开两个旋钮。
    6. ``HeadSpec(inputs=(HiddenStateRef(layer=-1),))`` →
       ``source=HiddenStateRef(stream=…)``；取最终输出是 ``layer=None``，
       ``-1`` 会和联合层号混淆。
    7. ``HeadSpec(inference=False)`` → ``gate=UniformSteps(skip={"video"})``。
       「推理期砍掉视频流」是步进策略而不是头的属性，否则同一事实有两个出处。
    8. ``Spec(ShiftedUniform, …)`` / ``Spec(Constant, …)`` →
       ``TimestepSampler(UniformShift(…))`` / ``TimestepSampler(Constant(…))``。
       分布是普通 dataclass；``Spec`` 只用在要进 ``nn.Module`` 的实现上。
    9. ``Spec(EulerSampler, {"num_steps": N})`` → ``EulerSampler()`` +
       ``UniformSteps(num_steps=N)``。步数归 ``SteppingGate``，与采样器解耦。

    ``config`` 形参是 ``build_world_model`` 的调用约定（``provider(config)``）。
    YAML 覆写的施加是 ``build_world_model`` 第③步、属配置层，尚未落地 ——
    所以这里对非 None 的 config 抛错而**不是**静默忽略它。
    """
    _reject_config("vpp2", config)

    # ① 流标量。未被消费的键抛错 —— spec_config §3.8 的第三条刻意严格。
    scalars = _resolve_scalars("vpp2", _VPP2_STREAM_SCALARS, stream_scalars)

    video, action = _dual_stream_wan_streams(
        video_impl=Spec(wan_video_stream),
        action_impl=Spec(action_stream),
        **scalars,
    )

    # ② 一个注意力组、组内两条流、无 KV 路由。
    #    联合层数 == 两条流各自的层数（ONE_TO_ONE，V2 在此解析）。
    #    ``routes=()`` 是刻意的：VPP2 默认配置下跨流信息走**联合注意力**本身
    #    （``mot.py:916-935`` 把两条流的 q/k/v 拼起来算一次再按 seq_lens split）。
    #    ActionDiT 里那套 ``video_kv_projections``（``action_dit.py:124-146``，
    #    动作流自带逐层 K/V 投影去读冻结的视频特征）才是 ``KVRoute`` 对应的东西，
    #    但它挂在 ``use_video_tokens`` 上、默认 False（``action_dit.py:67``），
    #    两份训练配置都没开 ⇒ 默认拓扑里没有这条边。
    group = AttentionGroup(members=(video, action),
                           num_layers=video.num_layers)

    # ③ L3。两条流的 timestep 分布不同，这是 ``timesteps`` 按流名索引的理由。
    diffusion = DiffusionSpec(
        timesteps={
            # video：``train_sampling_strategy: pure_noise``
            # （``train_action.yaml:204``，代码 ``vpp2.py:1608`` 把它钉死在
            # num_train_timesteps）⇒ t 恒为纯噪声端。用 ``Constant`` 而不是
            # 在流里写 ``if not add_noise``：后者让「这条流的噪声水平」在 spec
            # 里不可见，``noise_gate`` 就读不到它。
            # 同一节配的 ``beta_alpha: 7.0 / beta_beta: 1.0`` 在 pure_noise 下
            # **不参与采样**，所以这里不配 ``BetaDist`` —— 配了就等于凭空
            # 多出一个生效中的旋钮。
            "video": TimestepSampler(Constant(value=1.0)),
            # action：``train_shift: 5.0``（``train_action.yaml:211``），
            # 同节的 ``infer_shift`` 也是 5.0 —— 训练推理同值，一个分布位够用。
            # 视频流那边训练 5.0 / 推理 3.0 **不同值**，但它推理期被 ``gate``
            # 整条砍掉（下面 ``skip={"video"}``），所以那个 3.0 没有落点；
            # 真要表达「同一条流训练与推理两套分布」需要 DiffusionSpec 再开一位，
            # 等到有模型真的推理期也跑视频流时再加。
            "action": TimestepSampler(UniformShift(shift=action_shift)),
        },
        # ``lambda_video`` / ``lambda_action`` 都是 1.0（``train_action.yaml:217-218``）
        losses=(
            LossSpec(name="video_flow", head="video_flow"),
            LossSpec(name="action_flow", head="action_flow"),
        ),
        schedule=RectifiedFlow(),
        # ``skip={"video"}`` 就是 §4.4 样例里那个 ``HeadSpec(inference=False)``：
        # 「推理期砍掉视频流」是步进策略，不是头的属性。
        # 步数取 ``evaluation.num_inference_steps: 10``（``train_action.yaml:224``，
        # 与 ``policy/VPP2/deploy.yml:21`` 同值）。
        gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        sampler=EulerSampler(),
        # 两档：clean（条件帧）与 noisy。非空 ⇒ ``noise_gate`` 规则可用，
        # 这是下面 ``_vpp2_visibility`` 第二条规则的前提。
        noise_levels=(0.0, 1.0),
    )

    # ④ L4。两个头都吃各自流的**最终输出**（layer=None），都消费 timestep：
    #    ``Head.forward(x, t)``（``wan_video_dit.py:287-297``）与
    #    ``ActionHead.forward(x, t)``（``action_dit.py:26-34``）各自带一张
    #    ``(1, 2, dim)`` 的 modulation 参数 ⇒ 恰好是 ``HEAD_SLOTS`` 两槽。
    heads = (
        # 16 是 VAE 的 latent 通道数（``wan_video_dit.py:328`` 的 ``out_dim``）；
        # patch 反展开的 ``× prod(patch_size)`` 在头内部（``:286``），
        # 不进 ``output_width`` —— 否则同一个数会有两种含义。
        HeadSpec(name="video_flow", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="video"),
                 output_width=video_latent_channels,
                 needs_timestep=True),
        # 动作头的宽度是**打包后**的动作维 = raw_action_dim × action_chunk_size
        # （``vpp2.py:244`` 的反向；``action_dit.py:150`` 那个 Linear 的输出宽度）。
        # libero 是 7 × 1 = 7（``train_action.yaml:193`` / ``:166``）。
        HeadSpec(name="action_flow", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="action"),
                 output_width=action_width,
                 needs_timestep=True),
    )

    # ⑤ 条件。只声明已逐行核实的那一条：text cross-attn 的 KV，两条流都吃
    #    （video 侧 ``wan_video_dit.py:522`` 的 ``text_embedding``，
    #    action 侧 ``action_dit.py:405`` 同名）。
    #
    #    **记一笔，不静默**：现状往这同一个 KV 张量里还拼了两样东西 ——
    #      - CLIP 图像特征，只拼进 video 侧（``wan_video_dit.py:523-526``
    #        ``cat([image_context, text_context])``）；
    #      - proprio（8 维）过一个 Linear 成一个 token，拼进两条流共用的
    #        context（``vpp2.py:520-539``，调用点 ``:843`` / ``:1597``）。
    #    它们物理上**不是独立的槽**，而四个 ``InjectionKind`` 里没有
    #    「追加到已有 KV 槽」这一种。多声明两条 CROSS_ATTENTION 条件会被
    #    V7 要求两条流各多开一个 ``cross_kv_slots``，而那与代码不符。
    #    这一处的建模属 Lane B 的条件总线，留到 L2 落地时裁定。
    #
    #    i2v 的图像条件也没在这里声明：它既不是 cross-attn 也不是通道拼接，
    #    而是**沿时间轴**把干净的条件 latent 帧替换掉噪声序列的前若干帧
    #    （``vpp2.py:815-817``，``dim=2``）。这正是 ``visible_video_frames``
    #    与下面 noise_gate 规则所描述的那同一件事，声明成条件会变成第二个出处。
    conditions = (
        ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                      targets=("video", "action"), source_key="context"),
    )

    return WorldModelSpec(
        model_type="vpp2",
        # 几何取 libero 配置（``train_action.yaml:77-83``）：
        # ``num_frames: 65`` / ``action_horizon: 32`` /
        # ``action_video_freq_ratio: 4`` / ``video_size: [224, 448]`` ——
        # 448 是**拼接后**的宽，所以这里拆成 224 × ``num_views=2``
        # （``concat_multi_camera: horizontal``，``num_output_cameras: 2``）。
        #
        # ``vae_spatial_stride=8`` 必须显式给：VPP2 用的是 **Wan2.1 的 VAE**
        # （``wan_video_vae.py:648-650`` 的 ``upsampling_factor = 8`` /
        # ``temporal_downsample_factor = 4`` / ``z_dim = 16``），
        # 而 ``GeometrySpec`` 的缺省 16 是 Wan2.2 那一代（fastwam 的
        # ``WanVideoVAE38``）。``video_dit_config.out_dim: 16`` 是同一件事的
        # 另一个侧面 —— 16 通道 ⇒ Wan2.1 ⇒ 8 倍空间压缩。
        # 派生：latent (17, 28, 56) → patch (17, 14, 28) ⇒ 每帧 392 token、
        # 共 6664 token；``num_chunks = 32 / 4 = 8``。
        # 这些字段标 [YAML]（spec_config §3.3），覆写的施加是第③步。
        geometry=GeometrySpec(
            frame_height=224,
            frame_width=224,
            num_frames=65,
            num_views=2,
            vae_spatial_stride=8,
            action_horizon=32,
            action_video_freq_ratio=4,
        ),
        groups=(group,),
        diffusion=diffusion,
        heads=heads,
        conditions=conditions,
        visibility=_vpp2_visibility(visible_video_frames),
        # 现状是一张 dense 的 ``[S, S]`` bool mask 直接喂给注意力核
        # （``mot.py:144-157`` 的 ``flash_attention(..., ctx_mask=)``），
        # 所以默认后端如实写 ``dense``。换 ``flex`` 要同时给
        # ``attention_block_size``（V9），并且它必须与 ``geometry.align_to``
        # 对齐 —— align_to 的归属是总纲 §10-1 的未决项，换后端等那条裁定。
        attention_backend="dense",
    )


# ---- provider：fastwam（基线；族默认值就是它的实测取值）----

_FASTWAM_STREAM_SCALARS: dict[str, int] = {}
"""fastwam **一个流标量都不覆写**。

``_dual_stream_wan_streams`` 的十个默认值就是它的实测取值
（``model/fastwam/modeling_configuration_fastwam.py:75-96``：video
3072/24×128/30/14336、action 1024/24×128/30/4096）。空表是刻意的 ——
把同一组数再抄一遍就等于给同一事实开第二个出处，而族默认值本来就是
按「基线 fastwam」挑的。YAML 仍可逐键覆写（见 ``_resolve_scalars``）。
"""


def _fastwam_visibility(tokens_per_frame: int) -> dict[str, Visibility]:
    """fastwam 的 L1。一手事实 ``model/fastwam/mot/fastwam.py:433-456``。

    三次写入，与 VPP2 同形：video→video 来自 ``build_video_to_video_mask``、
    action→action 全 True（``:452``）、action→**首个 latent 帧**
    （``:455``，``first_frame_tokens = min(video_tokens_per_frame, video_seq_len)``）。
    同样**没有 video → action 这条边**。

    **两处与设计文档 §2 的 fastwam 行不一致，按代码写并在此记账：**
      1. §2 把 action 侧写成 ``same_stream() & chunk_causal()``；代码 ``:452``
         是全 True，没有任何块因果性。多那条规则会遮掉动作自己的未来块。
      2. §2 的 video 侧对应 ``first_frame_causal``，但**同名的预设与同名的代码
         不是一回事**：代码（``model/fastwam/wan/dit.py:595-600``）是
         「先全 True，再把首帧那几行指向后续帧的边打掉」——
         首帧之外的每一帧都是完全双向的，连未来帧都看得见；
         预设 ``frame_causal() | prefix_only(n)`` 则是逐帧因果。
         九条原子规则表达不出代码那个形状（它是「只有首帧受限」）。
         这里取**更严**的预设：等价性测试要的是「不多给可见性」，
         宁可过严不可过宽 —— 过宽会让模型在等价性测试里看到它本不该看的
         token 而数值仍然收敛，那种偏差查不出来。
    """
    first_frame_causal = get_preset("first_frame_causal")
    return {
        "video": first_frame_causal(prefix_tokens=tokens_per_frame),
        "action": same_stream() | (
            stream_edge("video", "action") & prefix_only(tokens_per_frame)
        ),
    }


@register_provider("fastwam")
def _fastwam(config: Any = None, *,
             video_shift: float = 5.0,
             action_shift: float = 5.0,
             inference_steps: int = 20,
             video_latent_channels: int = 48,
             action_width: int = 7,
             **stream_scalars: Any) -> WorldModelSpec:
    """``--model-name fastwam``。族里的基线成员，已在落地仓里。

    与 vpp2 的三处结构差异（都是数值/拓扑而非契约差异）：

    1. **video 的 timestep 不是常量**。fastwam 两条流都真采 t
       （``video_scheduler`` 与 ``action_scheduler`` 在
       ``modeling_configuration_fastwam.py:98-103`` 是**同一组数**：
       ``train_shift: 5.0`` / ``infer_shift: 5.0``）⇒ 两个
       ``UniformShift(5.0)``，不是 vpp2 那个 ``Constant(1.0)``。
       **设计文档 §2 的 fastwam 行写的是 video 3.0 / action 1.0，与代码不符**，
       按代码写并在此记一笔。
    2. **``seperated_timestep: True``**（``:82``）。它在代码里的实际形状是
       ``token_timesteps = ones(B, F, tokens_per_frame) * t.view(B,1,1)``
       然后 ``token_timesteps[:, 0, :] = 0``（``wan/dit.py:632-646``，
       非 separated 的那条分支 ``:647`` 直接抛 NotImplementedError）——
       **一次 ``[B]`` 采样广播到全部帧，再把首个 latent 帧钉成干净的**。
       所以它**不是** ``BroadcastSpec(axes=("frame",))``（那意味着 F 次独立采样）：
       逐流仍是默认的 per-sample 广播，「首帧干净」由 ``noise_levels`` 两档
       加上 L1 里那条 ``prefix_only(tokens_per_frame)`` 表达。
       采样本身 ``action/schedulers.py:51-58`` 也确实只出 ``[B]``。
    3. **单帧条件、两路相机**。``num_video_frames: 9`` 是**采样后**的帧数
       （``data_configuration_fastwam.py:64-73`` 的 ``observation_delta_indices``
       证明原始帧数 = ``(T-1) × ratio + 1``），VAE 看到的就是这 9 帧。

    ``inference_steps`` 默认 20 取代码签名（``mot/fastwam.py:982``）——
    仓里没有任何配置覆写它，与 vpp2 的 10 不同源也不同值。
    """
    _reject_config("fastwam", config)

    scalars = _resolve_scalars("fastwam", _FASTWAM_STREAM_SCALARS,
                               stream_scalars)
    video, action = _dual_stream_wan_streams(
        video_impl=Spec(wan_video_stream),
        action_impl=Spec(action_stream),
        **scalars,
    )
    group = AttentionGroup(members=(video, action),
                           num_layers=video.num_layers)

    # 几何取 ``configs/models/embodied/fastwam.yaml``：``num_video_frames: 9``
    # （:27）、``image_size: 224``（:28）、``action_horizon: 32``（:13）、
    # ``action_video_freq_ratio: 4``（:35）。两路相机沿**宽度**拼
    # （``fastwam_transform.py:57-65`` 的 ``cat(dim=-1)``，libero_2cam）。
    # VAE 用缺省的 16/4：fastwam 是 **Wan2.2 的 ``WanVideoVAE38``**
    # （``model/fastwam/wan/vae.py:1469-1499``：``z_dim=48`` /
    # ``upsampling_factor=16`` / ``temporal_downsample_factor=4``），
    # 与 vpp2 的 Wan2.1（8 倍、16 通道）**不同代**。
    # 派生：latent (3, 14, 28) → patch (3, 7, 14) ⇒ 每帧 98 token、
    # 共 294 token；``num_chunks = 32 / 4 = 8``。
    geometry = GeometrySpec(
        frame_height=224,
        frame_width=224,
        num_frames=9,
        num_views=2,
        action_horizon=32,
        action_video_freq_ratio=4,
    )

    diffusion = DiffusionSpec(
        timesteps={
            "video": TimestepSampler(UniformShift(shift=video_shift)),
            "action": TimestepSampler(UniformShift(shift=action_shift)),
        },
        # ``lambda_video`` / ``lambda_action`` 都是 1.0（``:104-106``）
        losses=(
            LossSpec(name="video_flow", head="video_flow"),
            LossSpec(name="action_flow", head="action_flow"),
        ),
        schedule=RectifiedFlow(),
        # 推理期只跑动作流：``infer_action``（``mot/fastwam.py:982``）先
        # ``mot.prefill_video_cache``（``:1081``）一次，之后每步只调
        # ``forward_action_with_video_cache``（``:774`` / ``:1106``）。
        gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        sampler=EulerSampler(),
        # 两档 = 首个 latent 帧 clean + 其余 noisy（``wan/dit.py:645``）。
        noise_levels=(0.0, 1.0),
    )

    heads = (
        # 48 是 Wan2.2 VAE 的 latent 通道数
        # （``modeling_configuration_fastwam.py:78`` 的 ``out_dim: 48``）。
        # 头内那个 Linear 实际宽 48 × prod(patch) = 192，反展开在头内部
        # ⇒ 不进 ``output_width``。
        HeadSpec(name="video_flow", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="video"),
                 output_width=video_latent_channels,
                 needs_timestep=True),
        # ``model/fastwam/action/dit.py:128`` 的 ``Linear(hidden_dim, action_dim)``
        # ⇒ 7（``action_dim: 7``，``fastwam.yaml:12``）。
        HeadSpec(name="action_flow", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="action"),
                 output_width=action_width,
                 needs_timestep=True),
    )

    return WorldModelSpec(
        model_type="fastwam",
        geometry=geometry,
        groups=(group,),
        diffusion=diffusion,
        heads=heads,
        # 与 vpp2 同形：逐层一个 text cross-attn 槽，两条流都吃。
        # proprio（``proprio_dim: 8``）同样是拼进这个 context 的一个 token，
        # 不是独立的槽 —— 同一处建模空白，留到 Lane B 条件总线裁定。
        conditions=(
            ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                          targets=("video", "action"), source_key="context"),
        ),
        visibility=_fastwam_visibility(geometry.tokens_per_frame),
        # ``mot/fastwam.py:433-456`` 产的是 dense ``[S, S]`` bool。
        attention_backend="dense",
    )


# ---- provider：internw0_delta（§4.6）----

_INTERNW0_DELTA_STREAM_SCALARS: dict[str, int] = {}
"""InternW0-Δ 的两条流与族默认值**逐字段相同**。

一手出处 ``projects/InternW0-Delta/configs/model/wam.yaml``：video
``hidden_dim 3072``(:21) / ``ffn_dim 14336``(:22) / ``num_heads 24``(:26) /
``attn_head_dim 128``(:27) / ``num_layers 30``(:28)；action
``hidden_dim 1024``(:44) / ``ffn_dim 4096``(:45) / ``num_heads 24``(:46) /
``attn_head_dim 128``(:47) / ``num_layers 30``(:48)。

**记一笔**：设计文档说 ``factory.py`` 会把动作流的注意力宽度改写成
``num_heads × attn_head_dim``。它不会 —— ``factory.py:101-110`` 只**校验**
两条流的头数与头维相等、不等就抛错。所以 action 流残差 1024 而注意力
24 × 128 = 3072 这件事在这里是**原样保留**的，正是 ``residual_width``
与注意力宽度必须分开两个字段的理由。
"""


def _internw0_delta_visibility(tokens_per_frame: int,
                               condition_frames: int) -> dict[str, Visibility]:
    """InternW0-Δ 的 L1。一手事实 ``frame_attention.py:6-56``。

    现状布局（单一相机、96 token/帧）：video 0–479 = anchor 0–95 +
    recent 96–191 + current-clean 192–287 + noisy-future 288–479；
    **delta 480–575**；action 576–607。

    代码的七次写入：``:27-31`` 三个条件帧逐帧因果；``:35``
    ``mask[288:480, :480] = True``（噪声未来对全部真实视频**完全双向**，
    既看不到 delta 也看不到 action）；``:38-46`` delta 读 ``[96, 288)``
    （recent + current-clean，**不含 anchor**）；``:47-50`` delta 读 delta；
    ``:52`` action 读 ``[0, 288)``（只有条件帧，**永远看不到噪声未来**）；
    ``:53-54`` action 读 delta；``:55`` action 读 action。

    **四处表达不出来，逐条记账（按「宁过严不过宽」裁定，不静默取一侧）：**
      1. **delta token 不是一条流**。那 96 个 token 是在 video 流内部追加的
         （``wan_video_dit.py:588-590`` 追加一次、``:959-967`` 末尾截掉），
         既不在 ``GeometrySpec`` 的 token 账里，也没有自己的 ``StreamSpec``。
         于是 delta 的三条规则与 action→delta 这条边**都没有落点**。
         §4.6.7 说它应当成为 ``GeometrySpec`` 的一个派生属性 —— 等那条裁定。
      2. **``frame_causal()`` 对噪声未来过严**。代码里 288–479 这些行是全双向的；
         逐帧因果会把它们限成只看过去。过严不过宽：等价性测试里「多看了
         不该看的 token」数值上仍会收敛，查不出来；少看了会立刻露出来。
      3. **action 侧只能用 ``prefix_only(3 × 96)``** 表达「只读三个条件帧」。
         它与 ``:52`` 的 ``[0, 288)`` 恰好同值，但这是因为条件帧正好在序列最前 ——
         一旦布局变了这两件事就分叉。``history(policy)`` 才是它的归处。
      4. 文档里的 ``sees`` / ``reads`` / ``delta_reads_local`` 这三个名字在
         ``visibility/rules.py`` 里**不存在**；九条原子规则里 ``frame_causal``
         的形参是 ``window`` 而不是 ``condition_frames``。
    """
    return {
        "video": same_stream() & frame_causal(),
        "action": same_stream() | (
            stream_edge("video", "action")
            & prefix_only(condition_frames * tokens_per_frame)
        ),
    }


@register_provider("internw0_delta")
def _internw0_delta(config: Any = None, *,
                    condition_frames: int = 3,
                    video_shift: float = 6.0,
                    action_shift: float = 1.0,
                    inference_steps: int = 10,
                    video_latent_channels: int = 48,
                    action_width: int = 80,
                    **stream_scalars: Any) -> WorldModelSpec:
    """``--model-name internw0_delta``。族里损失条数最多的那个（**5 条**）。

    两条流的装配与 fastwam 完全同值（见 ``_INTERNW0_DELTA_STREAM_SCALARS``），
    差异全在 L3/L4：五路监督 + 两个从**中间层**取隐藏态的对齐头。
    这正是 ``losses`` 必须是列表、``HiddenStateRef`` 必须带 ``layer`` 的理由。

    一手事实在 ``projects/InternW0-Delta/``，逐处标行号。
    """
    _reject_config("internw0_delta", config)

    scalars = _resolve_scalars("internw0_delta",
                               _INTERNW0_DELTA_STREAM_SCALARS, stream_scalars)
    video, action = _dual_stream_wan_streams(
        video_impl=Spec(wan_video_stream),
        action_impl=Spec(action_stream),
        **scalars,
    )
    group = AttentionGroup(members=(video, action),
                           num_layers=video.num_layers)

    # 几何。一手数值：``configs/data/pretrain.yaml:10`` ``video_size (384, 256)``
    # = (H, W)、``:33`` ``num_output_cameras: 1``、
    # ``configs/pretrain/dataset.yaml:320`` ``action_size 32`` /
    # ``:322`` ``action_video_freq_ratio 4``。VAE 同 fastwam 是 Wan2.2 的
    # ``WanVideoVAE38``（``wan_video_vae.py:1358-1387``，16/4/z_dim 48），
    # ``patch_size [1,2,2]``（``configs/model/wam.yaml:16-19``）。
    #
    # **``num_frames=17`` 不是配置里的任何一个数，这里记一笔**：
    # 配置是 ``num_frames 33``（``dataset.yaml:319``）、解出 9 帧 RGB
    # （``robot_video_dataset.py:130`` 的 ``range(0, 33, 4)``）⇒ 噪声窗口
    # 1 + (9-1)/4 = **3 个 latent 帧**。但进 MoT 的视频前缀是
    # ``cat([anchor_latents, recent_latents, noisy_window], dim=2)``
    # （``losses.py:598-600``）= anchor 1 + recent 1 + window 3 = **5 个
    # latent 帧** = 5 × 96 = **480 token**（``wan_video_dit.py:814``）。
    # ``GeometrySpec`` 目前**没有「另外拼几个条件 latent 帧」这个字段**，
    # 而 ``padded_seq_len`` 是静态形状的唯一来源（INV-2）⇒ 它必须等于 480。
    # 17 是「能压出 5 个 latent 帧」的等效压缩前帧数（1 + 16/4 = 5）。
    # 代价：``num_frames`` 在本 provider 里不再等于数据侧的帧数。
    # 正解是 §4.6.7 说的那个 —— 给 ``GeometrySpec`` 加条件帧数字段，
    # 让 17 这个凑出来的数消失。那属 Lane A。
    # 派生：latent (5, 24, 16) → patch (5, 12, 8) ⇒ 每帧 96 token、
    # 共 480 token；``num_chunks = 32 / 4 = 8``。
    geometry = GeometrySpec(
        frame_height=384,
        frame_width=256,
        num_frames=17,
        num_views=1,
        action_horizon=32,
        action_video_freq_ratio=4,
    )

    # L3。两条流各采一次 t，都是 ``UniformShift``，但 shift 不同值：
    # video 6.0（``wam.yaml:53-56``，train/infer 同值）、
    # action 1.0（``:57-60``，同样 train/infer 同值）。
    #
    # **``seperated_timestep: true``（``wam.yaml:30``）不是 per-frame 采样**。
    # ``losses.py:624-634`` 逐字核过：
    #   ``cat([zeros(B, num_condition_frames - 1),
    #          timestep_video[:, None].expand(B, future_latent_frames)], dim=1)``
    # ⇒ **一次 ``[B]`` 采样**广播到未来帧、条件帧钉 0。所以 broadcast 取默认的
    # per-sample，**不是**文档 §4.6.4 写的 ``sample_shape=("batch","frame")``
    # —— 那意味着 F 次独立采样，与代码不符。逐帧那层结构由 ``noise_levels``
    # 两档 + L1 里条件帧的前缀规则表达。
    # （``action_dit.py:348-392`` 那条 ``[B, T]`` 逐动作 token 的路径确实存在，
    # 但挂在 RTC 上、``wam.yaml:40-42`` ``enabled: false`` ⇒ 默认拓扑里不生效。）
    diffusion = DiffusionSpec(
        timesteps={
            "video": TimestepSampler(UniformShift(shift=video_shift)),
            "action": TimestepSampler(UniformShift(shift=action_shift)),
        },
        # 五条。权重与 ``losses.py:992-998`` 的
        # ``0.5·video + 1.0·action + 0.5·future_delta
        #   + 0.1·semantic_future + track_weight·track_raw`` 一一对应
        # （``wam.yaml:62`` / ``:63`` / ``:71`` / ``:76``）。
        # 这是 ``losses`` 必须是**列表**而不是「video/action 两个字段」的理由。
        losses=(
            LossSpec(name="video_flow", head="video_flow", weight=0.5),
            LossSpec(name="action_flow", head="action_flow", weight=1.0),
            LossSpec(name="future_delta", head="future_delta", weight=0.5),
            LossSpec(name="semantic_future", head="semantic_future",
                     weight=0.1),
            # track 的 0.1 是**上限**而不是常权：
            # ``track_bridge.py:157-163`` 的
            # ``weight = loss_weight * min(1.0, step / warmup_steps)``，
            # ``warmup_steps: 2000``（``configs/task/4D_distillation.yaml:28-30``）。
            # 这正是 I4′ ``WeightRamp`` 的落点 —— 模型自己不数步数，
            # step 由 trainer 从 ``batch["global_step"]`` 递进来。
            # 注意 ``LossSpec.weight_at`` 在有 ramp 时**忽略** ``weight``，
            # 所以终值写在 ``end_value`` 里而不是 ``weight`` 里。
            LossSpec(name="track_align", head="track_align",
                     ramp=WeightRamp(start_step=0, end_step=2000,
                                     start_value=0.0, end_value=0.1)),
        ),
        schedule=RectifiedFlow(),
        gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        sampler=EulerSampler(),
        # 两档 = 条件帧 clean + 未来帧 noisy（``losses.py:624-634`` 的 zeros 段）。
        noise_levels=(0.0, 1.0),
    )

    # L4。五个头，后三个是这个模型区别于 fastwam 的全部内容。
    #
    # **权重的归属**：``HeadSpec`` 也有一个 ``weight``（它是 §3.7 里唯一可
    # YAML 覆写的字段），这里**一律不用**，五个权重全写在 ``LossSpec`` 上。
    # 两处都能写同一个数是现有签名的一处冗余，记一笔：Lane C 落地时应当裁定
    # 只留一处，否则 ``0.5`` 会同时出现在两个对象里而只有一个生效。
    heads = (
        # 48 = ``out_dim``（``wam.yaml:25``）。头内 Linear 实宽
        # 48 × prod(patch) = 192（``wan_video_dit.py:406``）⇒ 不进 output_width。
        HeadSpec(name="video_flow", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="video"),
                 output_width=video_latent_channels,
                 needs_timestep=True),
        # ``action_dit.py:128`` 的 ``Linear(1024, 80)``（``action_dim 80``，
        # ``configs/data/pretrain.yaml:34``）。
        HeadSpec(name="action_flow", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="action"),
                 output_width=action_width,
                 needs_timestep=True),
        # future_delta：``Sequential(LayerNorm(3072), Linear(3072, 384))``
        # （``wan_video_dit.py:557-563`` / ``:657-667``）再 reshape 成
        # ``(2, 192)``，``patch_dim = 48 × 1 × 2 × 2 = 192``
        # （``world_action_model.py:248``），``future_delta_num_frames =
        # (33-1)//4//4 = 2``（``:243-247``）⇒ output_width 取 **192**（每帧宽），
        # 帧数 2 是另一个轴、不进这个字段。无调制 ⇒ needs_timestep=False。
        #
        # **记一笔**：这个头读的是 video 流里 ``[480:576]`` 那 96 个 delta token，
        # 而 ``HiddenStateRef`` **没有 ``segment`` 字段** ⇒ 「读哪一段」表达不出来，
        # 现在只能写成「吃 video 流的最终输出」。与上面几何那条记账同根
        # （delta token 没有自己的位置），一并等 Lane A 给 ``GeometrySpec``
        # 补条件帧/delta 字段后再收。
        HeadSpec(name="future_delta", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="video"),
                 output_width=192),
        # semantic_future：**没有投影层**，直接在 3072 维原始隐藏态上算
        # ``1 - cos``（``losses.py:58-72``）。student = **第 8 层**的 delta 段
        # ``[480:576]``，teacher = **第 20 层**的 ``[384:480]`` 且 detach
        # （``losses.py:959-976``，层号来自 ``wam.yaml:74-75``）。
        #
        # **记一笔，两处表达不出来**：① ``HeadSpec.source`` 是**单个** ref，
        # 装不下 teacher 那一侧；② 没有 ``detach`` 字段，而 teacher 必须 detach，
        # 否则这条自蒸馏会把梯度灌回第 20 层、变成另一个优化问题。
        # 这里只声明 student 侧，teacher 侧与 detach 留给 Lane C 裁定
        # （要么 ``HeadSpec`` 加一个 ``reference: HiddenStateRef | None``，
        # 要么这类对齐损失不走 ``HeadSpec`` 而另立一类）。
        HeadSpec(name="semantic_future", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="video", layer=8),
                 output_width=video.residual_width),
        # track_align：读**第 14 层**（``configs/task/4D_distillation.yaml:24``
        # 的 ``source_layer: 14``）→ ``video_to_track = Linear(3072, 1024)``
        # （``track_bridge.py:26``）→ ``student_projector`` 末端
        # ``Linear(2048, 1430)``（``:38-43``），``teacher_dim 1430``
        # （``4D_distillation.yaml:27``）⇒ output_width 取最终的 1430。
        HeadSpec(name="track_align", impl=Spec(flow_head),
                 source=HiddenStateRef(stream="video", layer=14),
                 output_width=1430),
    )

    return WorldModelSpec(
        model_type="internw0_delta",
        geometry=geometry,
        groups=(group,),
        diffusion=diffusion,
        heads=heads,
        # 逐层一个 text cross-attn 槽（``wan_video_dit.py:330-331``），两条流都吃。
        # **与 vpp2 的差别**：这里**没有 CLIP**
        # （``wam.yaml:31`` ``require_clip_embedding: false``，
        # ``wan_video_dit.py:472`` 断言它关着）。proprio 仍是拼进同一个 context
        # 的一个 token（``input_builder.py:550-555`` /
        # ``proprio_encoder.py:46-47`` 的 ``Linear(80, 4096)`` + ``cat(dim=1)``）
        # ⇒ 同样没有独立的槽可落，与 vpp2 是同一处建模空白。
        conditions=(
            ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                          targets=("video", "action"), source_key="context"),
        ),
        visibility=_internw0_delta_visibility(geometry.tokens_per_frame,
                                              condition_frames),
        # 族里唯一用 flex 的成员：``wam.yaml:9`` ``mot_attention_backend: flex``。
        # 块大小 **128**：``wam.yaml:10`` 写的是 64，但被
        # ``configs/task/pretrain.yaml:17`` 覆写成 128，而 mask 侧还另有一个
        # 硬编码的 ``self.flex_mask_block_size = 128``（``:113``）。
        # 三处同一个数、两个取值 —— 取 128（实际生效值），并记一笔：
        # 这正是 V9 要求「flex 必须显式给一个正的 block_size」的理由。
        # 另记：``mixture_of_transformers.py:593-603`` 有一条**静默**回退
        # （``force_sdpa = not bool(key_mask_all.all().item())``）——
        # 它既是一次 D2H 同步，也让「我在用哪个后端」在运行期不可知。
        attention_backend="flex",
        attention_block_size=128,
    )
