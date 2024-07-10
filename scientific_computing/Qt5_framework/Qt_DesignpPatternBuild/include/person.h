#ifndef PERSON_H
#define PERSON_H

#include <QObject>

class CPerson : public QObject
{
    Q_OBJECT
public:
    enum persontype{PRINCIPAL, TEACHER, STUDENT};
    Q_ENUM(persontype)

    // function member build returns a static pointer object of CPerson class type.
    // paramter is a r-value of enum persontype
    static CPerson * build(persontype type);
    persontype role();

    explicit CPerson(QObject *parent = nullptr);

signals:

public slots:
private:
    persontype m_type_;
};

#endif // PERSON_H
