{ 
    "FA:KE:4b:8d:49:68" : {
        "camera1_mac_address": {
            "serviceType": "find",
            "source": "file:///tmp/amlo.mp4",
            "enabled": true,
            "generalFaceDectDbFile": "found_service1.dat",
            "checkBlackList": true,
            "checkWhiteList": true,
            "ignorePreviousDb": true,
            "saveFacesDb": true
            },
        "camera2_mac_address": {
            "serviceType": "find",
            "source": "file:///tmp/amlo.mp4",
            "enabled": true,
            "generalFaceDectDbFile": "found_service2.dat",
            "checkBlackList": true,
            "checkWhiteList": true,
            "ignorePreviousDb": true,
            "saveFacesDb": true
            }
        },
    "9c:7b:ef:2a:b6:07" : {
        "camera11_mac_address": {
            "find": {
                "source": "file:///tmp/amlo.mp4",
                "enabled": false,
                "generalFaceDectDbFile": "found_service1.dat",
                "checkBlackList": true,
                "checkWhiteList": true,
                "ignorePreviousDb": true,
                "saveFacesDb": false 
                },
            "blackList": {
                "source": "file:///tmp/AmloPanaderia.mp4",
                "enabled": true
                },
            "ageAndGender": {
                "source": "rtsp://192.168.129.14:9000/live",
                "enabled": false,
                "generalFaceDectDbFile": "age_and_gender_service3.dat"
                },
            "recurrence": {
                "source": "file:///tmp/amlo.mp4",
                "cameraId": "camera2_mac_address ESTE SERVICIO FALLA YA QUE EJECUTA BLACKLIST instead of recurrence",
                "enabled": false,
                "checkBlackList": false,
                "blacklistDbFile": "edgarblack",
                "checkWhiteList": false
                }
            },
        "camera14_mac_address": {
            "blackList": {
                "source": "file:///tmp/AmloPanaderia.mp4",
                "enabled": true
                }
            }
        }
}
