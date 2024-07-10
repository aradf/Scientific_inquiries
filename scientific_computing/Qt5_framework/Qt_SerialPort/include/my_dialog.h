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

 #ifndef MY_DIALOG_H
 #define MY_DIALOG_H

 #include <QDialog>
 #include "serial_port.h"
 
class QDialogButtonBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTextEdit;
class QComboBox;
class QListWidget;

/*
 * MyDialog is a public child of QDialog class. 
 */
class MyDialog : public QDialog
{
    Q_OBJECT
public:
    /*
     * The MyDialog constructor takes a null pointer.  The parent is pointer variable.  It 
     * parent == 0x1234 that points to some memory location holding a variable of 
     * class type QQidget. *parent is the object of class type QWidget, &parent == 0xABCD.  
     */
    MyDialog(QWidget *parent = 0);
    ~MyDialog();

private slots:
    /* These private slots are function that consumes some signal connected later */
    void on_buttonPortsInfo_clicked();
    void enableGetDataButton();
    void on_buttonOpenPorts_clicked();
    void on_buttonSend_clicked();
    void read_serialData(QByteArray data);

private:
    /* private data members
     * host_label is a pointer variable that holds an address of memory block holding an 
     * instance of QLable class type.  host_label == 0x1234 is an l-value; *host_label is 
     * the instance of class type QLable (r-value).  &host_label == 0xABCD is the address
     * of the ponter variable itself.  The arrow -> operator could be used to invoke the
     * function members of host_label.
     */

    void load_ports();
    
    QLabel *host_label;
    QLabel *port_label;
    QLineEdit *host_lineEdit;
    QLineEdit *port_lineEdit;
    QLabel *status_label;
    QPushButton *ports_button;
    QPushButton *quit_button;
    QDialogButtonBox *button_box;

    QString current_fortune;
    quint16 block_size;

    QLabel *cmbports_label;
    QComboBox *cmb_ports;

    QPushButton *openport_button;

    QLabel *inMessage_label;
    QLineEdit *in_message;

    QPushButton *send_button;
    QLabel * listMessage_label;
    QListWidget *list_messages;
    CSerialPort _port;

 };

 #endif