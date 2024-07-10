QT -= gui

CONFIG += c++17 console
CONFIG -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
        main.cpp

qnx: target.path = /tmp/$$(TARGET)/bin
else: unix:!android: target.path = /opt/$$(TARGET)/bin
!isEmpty(target.path): INSTLL += target