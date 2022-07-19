#include "yolov7.h"

using namespace std;
using namespace cv; 

Yolo::Yolo(float prob, float nms)
{
    prob_threshold = prob;
    nms_threshold = nms;
}
void Yolo::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i=left;
    int j=right;
    float p = faceobjects[(left+right)/2].prob;
    while(i<=j)
    {
        if(faceobjects[i].prob>p)
        {
            i++;
        }
        if(faceobjects[j].prob<p)
        {
            j--;
        }
        if(i<=j)
        {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if(left<j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if(i<right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}
void Yolo::qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if(faceobjects.empty())
        return;
    qsort_descent_inplace(faceobjects, 0, faceobjects.size()-1);
}

float Yolo::sigmoid(float x)
{
    return static_cast<float>(1.f/(1.f + exp(-x)));
}
float Yolo::intersection_area(Object& a, Object& b)
{
    if(a.x>b.x+b.w || b.x>a.x+a.w || a.h>b.y+b.h || b.h>a.y + a.h)
    {
        return 0.f;
    }
    float area_w = min(a.x+a.w, b.x+b.w) - max(a.x, b.x);
    float area_h = min(a.y+a.h, b.y+b.h) - max(a.y, b.y);
    return area_w * area_h;
}

void Yolo::nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked)
{
    picked.clear();
    int n = faceobjects.size();
    std::vector<float> area(n);
    for(int i=0; i<n; i++)
    {
        area[i] = faceobjects[i].w * faceobjects[i].h;
    }
    for(int i=0; i<n; i++)
    {
        Object& a = faceobjects[i];
        int keep=1;
        for(int j=0; j<(int)picked.size();j++)
        {
            Object& b = faceobjects[picked[j]];
            float insert_area = intersection_area(a, b);
            float union_area = area[i] + area[picked[j]] - insert_area;
            float iou = insert_area / union_area;
            if(iou>nms_threshold)
            {
                keep=0;
            }
        }
        if(keep)
        {
            picked.push_back(i);
        }
    }
}

void Yolo::generate_proposals(ncnn::Mat& anchors, ncnn::Mat& in_pad, ncnn::Mat& feat_blob, int stride, std::vector<Object>& objects)
{
    int num_grid_x = feat_blob.w;
    int num_grid_y = feat_blob.h;

    int num_anchor = anchors.w / 2;
    int num_class = feat_blob.c / num_anchor - 5;
    // std::cout<<num_anchor<<std::endl;
    int feat_offset = num_class + 5;
    for (int q=0;q<num_anchor; q++)
    {
        float anchor_w = anchors[q*2];
        float anchor_h = anchors[q*2+1];
        for(int i=0; i<num_grid_y; i++)
        {
            for(int j=0; j<num_grid_x; j++)
            {
                int class_index = 0;
                float class_score = -FLT_MAX;
                for(int k=0; k<num_class; k++)
                {
                    float score = feat_blob.channel(q*feat_offset+5+k).row(i)[j];
                    //std::cout<<score<<std::endl;
                    if(score>class_score)
                    {
                        class_score = score;
                        class_index = k;
                    }
                }
                float box_score = feat_blob.channel(q*feat_offset+4).row(i)[j];
                float confidence = sigmoid(box_score)*sigmoid(class_score);
                // std::cout<<"conf:"<<confidence<<std::endl;
                // std::cout<<"pro:"<<prob_threshold<<std::endl;
                if(confidence>=prob_threshold)
                {
                    float bx = sigmoid(feat_blob.channel(q*feat_offset).row(i)[j]);
                    float by = sigmoid(feat_blob.channel(q*feat_offset+1).row(i)[j]);
                    float bw = sigmoid(feat_blob.channel(q*feat_offset+2).row(i)[j]);
                    float bh = sigmoid(feat_blob.channel(q*feat_offset+3).row(i)[j]);

                    float pb_cx = (bx*2.f-0.5f+j)*stride;
                    float pb_cy = (by*2.f-0.5f+i)*stride;
                    float pb_cw = pow(bw*2.f, 2)*anchor_w;
                    float pb_ch = pow(bh*2.f, 2)*anchor_h;

                    float x0 = pb_cx-pb_cw*0.5f;
                    float y0 = pb_cy-pb_ch*0.5f;
                    float x1 = pb_cx+pb_cw*0.5f;
                    float y1 = pb_cy+pb_ch*0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;
                    objects.push_back(obj);

                }
            }
        }
    }
    // std::cout<<objects.size()<<std::endl;

}