#ifndef CLUSTERS_H
#define CLUSTERS_H

#include <stdint.h>
//#include "clusters.h"

#define PIXDEPTH 10

//struct to store coordinates of a single pixel
typedef struct {
    uint16_t x; 
    uint16_t y;
    } Point;

//struct to represent a single cluster
typedef struct{
    uint64_t size; //number of pixels in cluster
    uint64_t value;//total signal in cluster
    // sum of x coordinates of constituent pixels weighted by their values
    uint64_t weightedx; 
    // sum of y coordinates of constituent pixels weighted by their values
    uint64_t weightedy;

} Cluster;

//struct to store many clusters
typedef struct{
    int num; //number of Cluster structs already sotred in the array
    int maxsize; //max size of the array
    Cluster** array; 
} CCollection;

void searchFrame(CCollection *collectionptr, uint16_t *arr, int n, int m, int  threshold);

void searchFrame_array(uint16_t *declustered, uint16_t *arr, int n, int m, int threshold);

#endif 
