# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""
HiRobotApi
"""
import urllib2


def pushInfo(info, toid='4249864', base_url='http://apiin.im.baidu.com/api/msg/groupmsgsend', 
        Hi_Robot_Access_Token='d0f21c5bc973f3e8f14ebb268ab5cd223'):
    """
    pushInfo
    """
    data_url = 'access_token=%s' % (Hi_Robot_Access_Token)
    url = base_url + '?' + data_url
    print url
    info = '{"message": {"header": {"toid": [%s]}, "body": [{"type": "MD", "content": \"%s\"}]}}' % (toid, info) 
    if isinstance(info, bytes) or isinstance(info, bytearray):
        info = unicode(info, "utf-8").encode('utf8')
    elif isinstance(info, unicode):
        info = info.encode('utf8')
    print info
    req = urllib2.Request(url=url, data=info)
    req.add_header('Content-Type', 'application/json')
    req.add_header('charset', 'utf-8')
    print urllib2.urlopen(req).read()

def pushATInfo(info, toid='4249864', base_url='http://apiin.im.baidu.com/api/msg/groupmsgsend', 
        Hi_Robot_Access_Token='d0f21c5bc973f3e8f14ebb268ab5cd223', atuserid='', faq_url=''):
    """
    pushInfo
    """
    data_url = 'access_token=%s' % (Hi_Robot_Access_Token)
    url = base_url + '?' + data_url
    print url
    info = '''
        {
            "message": {
                "header": {
                    "toid": [%s]
                },
                "body": [
                    {
                        "type": "TEXT",
                        "content": \"%s\"
                    },
                    {
                        "atuserids":[\"%s\"],
                        "atall":false,
                        "type":"AT"
                    }
                ]
            }
        }
    ''' % (toid, info, atuserid)
    info = info.encode('utf8')
    print info
    req = urllib2.Request(url=url, data=info)
    req.add_header('Content-Type', 'application/json')
    req.add_header('charset', 'utf-8')
    print urllib2.urlopen(req).read()

if __name__ == '__main__':
    pushInfo('test', toid='4249864')

