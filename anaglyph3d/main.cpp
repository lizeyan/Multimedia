#include <QColor>
#include <QImage>
#include <QString>
#include <cassert>
#include <QDir>
#include <iostream>

int main(int argc, char *argv[])
{
    QImage left("left.jpg");
    QImage right("right.jpg");
    for (auto &f : QDir::current().entryList())
        std::cout << f.toStdString() << std::endl;
    assert(left.width () == right.width () && left.height () == right.height ());
    for (int i = 0; i < left.width (); ++i)
        for (int j = 0; j < left.height (); ++j)
        {
            QColor _pixel = left.pixelColor (i, j);
            _pixel.setRed (0);
            left.setPixelColor (i, j, _pixel);
        }
    for (int i = 0; i < left.width (); ++i)
        for (int j = 0; j < left.height (); ++j)
        {
            QColor _pixel = left.pixelColor (i, j);
            _pixel.setRed (right.pixelColor (i, j).red ());
            left.setPixelColor (i, j, _pixel);
        }
    left.save ("anaglyph3d.jpg");
    return 0;
}
