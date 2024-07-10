#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>

#include "../include/serial_port.h"


CSerialPort::CSerialPort(QObject * parent) : QObject(parent)
{
    qInfo() << this << "Constructor: CSerialPort invoked ...";
}

CSerialPort::~CSerialPort()
{
    qInfo() << "Destructor: CSerialPort invoked ...";
    if (_serial_port != nullptr)
    {
        _serial_port->close();
        delete _serial_port;
    }

}

bool CSerialPort::connect(QString port_name)
{
    if (_serial_port != nullptr)
    {
        _serial_port->close();
        delete _serial_port;
    }
    _serial_port = new QSerialPort(this);
    // _serial_port->setPortName(cmb_ports->currentText());
    _serial_port->setPortName(port_name);
    _serial_port->setBaudRate(QSerialPort::Baud115200);
    _serial_port->setDataBits(QSerialPort::Data8);
    _serial_port->setParity(QSerialPort::NoParity);
    _serial_port->setStopBits(QSerialPort::OneStop);
    if (_serial_port->open(QIODevice::ReadOnly))
    {
        QObject::connect(_serial_port,
                &QSerialPort::readyRead,
                this,
                &CSerialPort::read_serialData);
    }

    return _serial_port->isOpen();
}

void CSerialPort::read_serialData()
{
  if(_serial_port->isOpen())
    emit data_recieved(_serial_port->readAll());
}

qint64 CSerialPort::write(QByteArray data)
{
    return qint64(-1);
    if(!_serial_port->isOpen())
    {
        return -1;
    }
    else
        {
           //_serial_port->write(in_message->text().toUtf8());   
           return _serial_port->write(data);   
        }
}

void CSerialPort::serialport_info()
{
    const auto serialPortInfos = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo &portInfo : serialPortInfos) 
    {
        qDebug() << "\n"
                 << "Port:" << portInfo.portName() << "\n"
                 << "Location:" << portInfo.systemLocation() << "\n"
                 << "Description:" << portInfo.description() << "\n"
                 << "Manufacturer:" << portInfo.manufacturer() << "\n"
                 << "Serial number:" << portInfo.serialNumber() << "\n"
                 << "Vendor Identifier:"
                 << (portInfo.hasVendorIdentifier()
                     ? QByteArray::number(portInfo.vendorIdentifier(), 16)
                     : QByteArray()) << "\n"
                 << "Product Identifier:"
                 << (portInfo.hasProductIdentifier()
                     ? QByteArray::number(portInfo.productIdentifier(), 16)
                     : QByteArray());
    }
}