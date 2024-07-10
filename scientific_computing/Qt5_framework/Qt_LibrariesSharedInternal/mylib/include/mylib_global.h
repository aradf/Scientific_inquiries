#ifndef MYLIB_GLOBAL_H
#define MYLIB_GLOBAL_H

#include <QtCore/qglobal.h>

/**
 * Q_DECL are symbols (class definitions) and must be exported 
 * and/or imported. 
 * Note: 'MYLIBSHARED_EXPORT' is used to define mylib class
 *       in 'mylib.h' file.
 */ 


#if defined(MYLIB_LIBRARY)
#  define MYLIBSHARED_EXPORT Q_DECL_EXPORT
#else
#  define MYLIBSHARED_EXPORT Q_DECL_IMPORT
#endif

#endif // MYLIB_GLOBAL_H
