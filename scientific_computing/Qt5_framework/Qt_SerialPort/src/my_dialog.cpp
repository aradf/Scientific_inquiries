/****************************************************************************
 **
 ** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
 ** Contact: http://www.qt-project.org/legal
 **
 ** This file is part of the examples of the Qt Toolkit.
 **
 ** $QT_BEGIN_LICENSE:BSD$
 ** You may use this file under the terms of the BSD license as follows:
 **
 ** "Redistribution and use in source and binary forms, with or without
 ** modification, are permitted provided that the following conditions are
 ** met:
 **   * Redistributions of source code must retain the above copyright
 **     notice, this list of conditions and the following disclaimer.
 **   * Redistributions in binary form must reproduce the above copyright
 **     notice, this list of conditions and the following disclaimer in
 **     the documentation and/or other materials provided with the
 **     distribution.
 **   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
 **     of its contributors may be used to endorse or promote products derived
 **     from this software without specific prior written permission.
 **
 **
 ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 ** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 ** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 ** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 ** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 ** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 ** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 **
 ** $QT_END_LICENSE$
 **
 ****************************************************************************/

#include <QtWidgets>
#include <QtNetwork>
#include <QSerialPortInfo>
#include <QSerialPort>

#include "../include/my_dialog.h"

/*
 * MyDialog Constructor:
 * pointer variable parent is passed on the parent class 'QDialog'
 */
MyDialog::MyDialog(QWidget *parent) : QDialog(parent)
{
    /*
     * allocate memory for the poiner variable host_label
     */
    host_label = new QLabel(tr("&Server name:"));
    port_label = new QLabel(tr("&Server port:"));
    cmbports_label = new QLabel(tr("&Combo port"));
    inMessage_label = new QLabel(tr("&In Message"));
    listMessage_label = new QLabel(tr("&List Message"));

    QString ip_address = "";

    if (ip_address.isEmpty())
        ip_address = QHostAddress(QHostAddress::LocalHost).toString();

    host_lineEdit = new QLineEdit(ip_address);
    port_lineEdit = new QLineEdit();
    port_lineEdit->setValidator(new QIntValidator(1, 65535, this));
    cmb_ports = new QComboBox();
    in_message = new QLineEdit();
    list_messages = new QListWidget();

    host_label->setBuddy(host_lineEdit);
    port_label->setBuddy(port_lineEdit);
    cmbports_label->setBuddy(cmb_ports);
    inMessage_label->setBuddy(in_message);
    listMessage_label->setBuddy(list_messages);
        
    status_label = new QLabel(tr("This examples requires that you run the "
                                 "Fortune Server example as well."));

    ports_button = new QPushButton(tr("Ports"));
    ports_button->setDefault(true);
    ports_button->setEnabled(false);

    quit_button = new QPushButton(tr("Quit"));

    openport_button = new QPushButton(tr("Open Ports"));
    openport_button->setDefault(true);
    openport_button->setEnabled(true);

    send_button = new QPushButton(tr("Send Message"));
    send_button->setDefault(true);
    send_button->setEnabled(true);

    button_box = new QDialogButtonBox;
    button_box->addButton(ports_button, QDialogButtonBox::ActionRole);
    button_box->addButton(quit_button, QDialogButtonBox::RejectRole);
    button_box->addButton(openport_button, QDialogButtonBox::ActionRole);
    button_box->addButton(send_button, QDialogButtonBox::ActionRole);

    /*
     * connect the host_lineEdit variable with a text changed signal to this pointer and enableGetDataButton slot.
     */   
    connect(host_lineEdit, 
            SIGNAL(textChanged(QString)),
            this, 
            SLOT(enableGetDataButton()));

    connect(port_lineEdit, 
            SIGNAL(textChanged(QString)),
            this, 
            SLOT(enableGetDataButton()));

    /*
     * connect the ports_button clicked signal to the request_portsInfo slot.
     */   
    connect(ports_button, 
            SIGNAL(clicked()),
            this, 
            SLOT(on_buttonPortsInfo_clicked()));

    connect(quit_button, 
            SIGNAL(clicked()), 
            this, 
            SLOT(close()));
    
    connect(openport_button,
            SIGNAL(clicked()),
            this,
            SLOT(on_buttonOpenPorts_clicked()));

    connect(send_button,
            SIGNAL(clicked()),
            this,
            SLOT(on_buttonSend_clicked()));

    QGridLayout *mainLayout = new QGridLayout;
    mainLayout->addWidget(host_label, 0, 0);
    mainLayout->addWidget(host_lineEdit, 0, 1);
    mainLayout->addWidget(port_label, 1, 0);
    mainLayout->addWidget(port_lineEdit, 1, 1);
    mainLayout->addWidget(status_label, 2, 0, 1, 2);
    mainLayout->addWidget(button_box, 3, 0, 1, 2);
    mainLayout->addWidget(cmbports_label, 4, 0);
    mainLayout->addWidget(cmb_ports, 4, 1);
    mainLayout->addWidget(inMessage_label, 5, 0);
    mainLayout->addWidget(in_message, 5, 1);
    mainLayout->addWidget(listMessage_label, 6, 0);
    mainLayout->addWidget(list_messages, 6, 1);

    setLayout(mainLayout);

    setWindowTitle(tr("Serial Communication"));
    port_lineEdit->setFocus();
    ports_button->setEnabled(false);
    status_label->setText(tr("Opening Serial session."));

    load_ports();
    QObject::connect(&_port,
                     &CSerialPort::data_recieved,
                     this,
                     &MyDialog::read_serialData);

}

MyDialog::~MyDialog()
{

}

void MyDialog::enableGetDataButton()
{
    qInfo() << host_lineEdit->text();
    qInfo() << port_lineEdit->text();
    qInfo() << in_message->text();
    qInfo() << "BaudRate: " <<  QSerialPort::Baud115200;
    qInfo() << "DataBits: " <<  QSerialPort::Data8;
    qInfo() << "Parity: " <<  QSerialPort::NoParity;
    qInfo() << "StopBits: " <<  QSerialPort::OneStop;

    ports_button->setEnabled(!host_lineEdit->text().isEmpty() &&
                             !port_lineEdit->text().isEmpty() );

}

void MyDialog::on_buttonPortsInfo_clicked()
{
    ports_button->setEnabled(false);
    _port.serialport_info();

    block_size = 0;
}

void MyDialog::on_buttonOpenPorts_clicked()
{
   auto is_connected = _port.connect(cmb_ports->currentText());
   if (!is_connected)
   {
      QMessageBox::critical(this,"Error", "There is a problem with connection ...");
   }
}

void MyDialog::load_ports()
{
    const auto serialPortInfos = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo &portInfo : serialPortInfos) 
    {
        qDebug() << "Port:" << portInfo.portName() << "\n";
        cmb_ports->addItem( portInfo.portName());
    }
}

void MyDialog::on_buttonSend_clicked()
{
    qInfo() << "Sending: " << in_message->text();
    auto num_bytes = _port.write(in_message->text().toUtf8());
    send_button->setEnabled(true);
}

void MyDialog::read_serialData(QByteArray data)
{
   QString data_string = QString(data);
   
   for(int i = 0; i < 10; i++)
   {
      list_messages->addItem(data_string);
   }
   
   qInfo() << "Receiving: " << data_string;
}