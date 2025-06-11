#ifndef __TYPES_H__
#define __TYPES_H__

#include <string>


struct Detection
{
    
    float bbox[4];  // x1, y1, x2, y2
    float conf;
    int class_id;
    Detection(): bbox{0.0f, 0.0f, 0.0f, 0.0f}, conf(0.0f), class_id(0) {}
    Detection(const Detection& other): bbox{other.bbox[0], other.bbox[1], other.bbox[2], other.bbox[3]}, conf(other.conf), class_id(other.class_id) {}
};

#endif  // __TYPES_H__
