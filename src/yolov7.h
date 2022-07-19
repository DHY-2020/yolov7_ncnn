#include<iostream>
#include <opencv2/opencv.hpp>
#include "net.h"


struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

class Yolo
{
public:
    Yolo(float prob, float nms);
    //~Yolo();
    void generate_proposals(ncnn::Mat& anchors, ncnn::Mat& in_pad, ncnn::Mat& feat_blob, int stride, std::vector<Object>& objects);
    void qsort_descent_inplace(std::vector<Object>& faceobjects);
    void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked);
private:
    float prob_threshold;
    float nms_threshold;
    float sigmoid(float x);
    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
    float intersection_area(Object& a, Object& b);

};