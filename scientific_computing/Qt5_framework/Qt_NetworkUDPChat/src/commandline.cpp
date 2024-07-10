#include "commandline.h"

CCommandLine::CCommandLine(QObject *parent, FILE *file_handle) : QObject(parent), stream_input(file_handle)
{

}

void CCommandLine::monitor()
{
    // loop forever;  Bad programming;
    while (true)
    {
        QString value = stream_input.readLine();
        emit command(value);
    }
}

