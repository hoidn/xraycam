#include "clusters.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
//#include "queue.h"
#include "arrayqueue.h"



void createPoint(Point *point, int i, int j) {
    point -> x = i; 
    point -> y = j; 
}

/*
insert a Cluster* into the array of a CCollection, allocating more memory
if necessary
*/
void insert(CCollection* coll, Cluster* cluster) {
    if (coll -> maxsize <= coll -> num) {
        int newsize = 2 * coll -> maxsize;
        coll -> array = (Cluster **) realloc(coll -> array, newsize * sizeof(Cluster*)); 
        coll -> maxsize = newsize; 
    }
    coll -> num += 1; 
    coll -> array[coll -> num - 1] = cluster; 
}
/*
From starting coordinates i and j in the frame, find the cluster composed of 
all connected above-threshold pixels
pre: frame[i][j] > threshold
n, m: dimensions of the array
i, j: coords of current pixel
*/
Cluster* searchClust(uint16_t *frame, Cluster *cluster, uint8_t *explored, int n, int m, int i, int j, int threshold){
    uint32_t itop, ibottom, ileft, iright; 
    //queue to store pixels that need to be explored
    //Queue queue = createQueue();
    cluster -> size = 0; 
    cluster -> value = 0;
    cluster -> weightedx = 0;
    cluster -> weightedy = 0;

    Point point;
    createPoint(&point, i, j); 
    push(&point); 
    //mark point as explored
    explored[i * m + j] = 1; 
    while (size() > 0) {
        //dequeue and update our coordinates 
        point = *pop();
        i = point.x; 
        j = point.y;

        itop = ((uint32_t ) i) * m + ((uint32_t) j) + 1;
        ibottom = ((uint32_t) i) * m + ((uint32_t) j) - 1;
        ileft = (((uint32_t) i) - 1) * m + ((uint32_t) j);
        iright = (((uint32_t) i) + 1) * m + ((uint32_t) j);

        //update cluster size and value
        cluster -> size += 1;
        cluster -> value += (uint32_t) frame[i * m  + j];
        cluster -> weightedx += i * frame[i * m + j];
        cluster -> weightedy += j * frame[i * m + j];

        //add above-threshold adjacent pixels to the queue
        if ((j < m - 1) && frame[itop] > threshold && 
                explored[itop] == 0) {
            point.x = i;
            point.y = j + 1;
            push(&point);
            explored[itop] = 1; 
       }
        if ((j > 0) && frame[ibottom] > threshold && 
                explored[ibottom] == 0) {
            point.x = i;
            point.y = j - 1;
            push(&point);
            explored[ibottom] = 1; 
        }   
        if ((i > 0) && frame[ileft] > threshold && 
                explored[ileft] == 0) {
            point.x = i - 1;
            point.y = j;
            push(&point);
            explored[ileft] = 1; 
        }
        if ((i < n - 1) && frame[iright] > threshold && 
                explored[iright] == 0) {
            point.x = i + 1;
            point.y = j;
            push(&point);
            explored[iright] = 1; 
        }
    }
    return cluster; 
}
    
                
Cluster* searchClust_8(uint8_t *frame, Cluster *cluster, uint8_t *explored, int n, int m, int i, int j, int threshold){
    uint32_t itop, ibottom, ileft, iright; 
    //queue to store pixels that need to be explored
    //Queue queue = createQueue();
    cluster -> size = 0; 
    cluster -> value = 0;
    cluster -> weightedx = 0;
    cluster -> weightedy = 0;

    Point point;
    createPoint(&point, i, j); 
    push(&point); 
    //mark point as explored
    explored[i * m + j] = 1; 
    while (size() > 0) {
        //dequeue and update our coordinates 
        point = *pop();
        i = point.x; 
        j = point.y;

        itop = ((uint32_t ) i) * m + ((uint32_t) j) + 1;
        ibottom = ((uint32_t) i) * m + ((uint32_t) j) - 1;
        ileft = (((uint32_t) i) - 1) * m + ((uint32_t) j);
        iright = (((uint32_t) i) + 1) * m + ((uint32_t) j);

        //update cluster size and value
        cluster -> size += 1;
        cluster -> value += frame[i * m  + j];
        cluster -> weightedx += i * frame[i * m + j];
        cluster -> weightedy += j * frame[i * m + j];

        //add above-threshold adjacent pixels to the queue
        if ((j < m - 1) && frame[itop] > threshold && 
                explored[itop] == 0) {
            point.x = i;
            point.y = j + 1;
            push(&point);
            explored[itop] = 1; 
       }
        if ((j > 0) && frame[ibottom] > threshold && 
                explored[ibottom] == 0) {
            point.x = i;
            point.y = j - 1;
            push(&point);
            explored[ibottom] = 1; 
        }   
        if ((i > 0) && frame[ileft] > threshold && 
                explored[ileft] == 0) {
            point.x = i - 1;
            point.y = j;
            push(&point);
            explored[ileft] = 1; 
        }
        if ((i < n - 1) && frame[iright] > threshold && 
                explored[iright] == 0) {
            point.x = i + 1;
            point.y = j;
            push(&point);
            explored[iright] = 1; 
        }
    }
    return cluster; 
}

/*
Perform cluster search. 
collectionptr: pointer to CCollection to store the clusters
arr: the frame data
n, m: frame dimensions
threshold: noise threshold
*/
void searchFrame(CCollection *collectionptr, uint16_t *arr, int n, int m, int threshold) {
    //TODO: set pixels on the 'rim' equal to 0 (in main)
    //boolean array that record which pixels have been explored by the BFS. 
    Cluster* clusterptr = (Cluster *) malloc(sizeof(Cluster));
    uint8_t explored[n * m * sizeof(uint8_t)];
    memset(&explored, 0, n * m * sizeof(uint8_t));
    //uint8_t *explored = calloc(n * m, sizeof(uint8_t));
    for (int i = 1; i < n - 1; i ++) {
        for (int j = 1; j < m - 1; j ++) {
            if (explored[i * m + j] == 0 && arr[i * m + j] > (uint16_t) threshold) {
                searchClust(arr, clusterptr, &explored[0], n, m, i, j, threshold); 
                insert(collectionptr, clusterptr); 
            }
        }
    }
}


/*
Perform cluster search, taking an array instead of a CCollection*
collectionptr: pointer to CCollection to store the clusters
arr: the frame data
n, m: frame dimensions
threshold: noise threshold

Writes a 'declustered' array in which the signal from each
cluster is concentrated in one pixel to *declustered.
*/
void searchFrame_array(uint16_t *declustered, uint16_t *arr, int n, int m, int threshold) {
    InitQueue();
    Cluster cluster; 
    //boolean array that records which pixels have been explored by the BFS. 
    uint8_t explored[n * m * sizeof(uint8_t)];
    memset(&explored, 0, n * m * sizeof(uint8_t));
    for (int i = 1; i < n - 1; i ++) {
        for (int j = 1; j < m - 1; j ++) {
            if (explored[i * m + j] == 0 && arr[i * m + j] > (uint16_t) threshold) {
                searchClust(arr, &cluster, &explored[0], n, m, i, j, threshold); 
                declustered[(cluster.weightedx * m + cluster.weightedy)/cluster.value] = cluster.value;
            }
        }
    }
}

void searchFrame_array_8(uint32_t *declustered, uint8_t *arr, int n, int m, int threshold) {
    InitQueue();
    Cluster cluster; 
    //boolean array that records which pixels have been explored by the BFS. 
    uint8_t explored[n * m * sizeof(uint8_t)];
    memset(&explored, 0, n * m * sizeof(uint8_t));
    for (int i = 1; i < n - 1; i ++) {
        for (int j = 1; j < m - 1; j ++) {
            if (explored[i * m + j] == 0 && arr[i * m + j] > (uint8_t) threshold) {
                searchClust_8(arr, &cluster, &explored[0], n, m, i, j, threshold); 
                declustered[(cluster.weightedx * m / cluster.value)+
                    (cluster.weightedy/cluster.value)] = cluster.value;
            }
        }
    }
}
