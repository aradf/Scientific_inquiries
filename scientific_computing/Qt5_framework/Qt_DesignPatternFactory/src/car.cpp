#include "car.h"
#include "slow.h"
#include "sports.h"
#include "race.h"

CCar::CCar(QObject *parent) : QObject(parent)
{

}

CCar *CCar::make(CCar::Model model)
{
    QMetaEnum metaEnum = QMetaEnum::fromType<CCar::Model>();
    qInfo() << "Creating: " << metaEnum.valueToKey(model);

    switch (model) 
    {
        case CCar::Model::SLOWCAR:
            return new slow(nullptr);
        case CCar::Model::SPORTSCAR:
            return new sports(nullptr);
        case CCar::Model::RACECAR:
            return new race(nullptr);
    }

    return new slow(nullptr);
}
