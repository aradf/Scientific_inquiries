#include "../include/mythread.h"

MyThread::MyThread(int ID, QObject *parent) : QThread(parent)
{
   this->socketDescriptor = ID;

}

void MyThread::run()
{
   //thread starts hear.
   qDebug() << "Starting Thread";
   socket = new QTcpSocket();
   if (!socket->setSocketDesciptor(this->socket_descriptor))
   {
      emit error(socket->error());
      return;
   }
   
}