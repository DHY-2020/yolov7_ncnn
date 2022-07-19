
#include "yolov7.h"
#include <chrono>
#include <classnames.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

int main()
{
    // std::string model_param = "yolov5s.param";
    // std::string model_bin = "yolov5s.bin";
    ncnn::Net net;
    net.load_param( "yolov7-tiny-opt.param");
    net.load_model("yolov7-tiny-opt.bin");
    string img_path = "Cat.jpg";
    cv::Mat src_img = cv::imread(img_path);
    int tartget_size = 640;
    int height = src_img.rows;
    int width = src_img.cols;
    float scale = 1.f;
    int h,w;
    if(height>width)
    {
        scale = (float)tartget_size/height;
        h = tartget_size;
        w = width*scale;

    }
    else
    {
        scale = (float)tartget_size/width;
        w = tartget_size;
        h = height*scale;
    }
    int pad_h = (tartget_size - h)/2;
    int pad_w = (tartget_size-w)/2;
 
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(src_img.data, ncnn::Mat::PIXEL_BGR2RGB, src_img.cols, src_img.rows, w, h);
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, pad_h, pad_h, pad_w, pad_w, 0, 114.f);
    float norm[3] = {1/255.f, 1/255.f, 1/255.f};
    in_pad.substract_mean_normalize(0, norm);
    // auto start_time = system_clock::now();
    ncnn::Extractor ex = net.create_extractor();
    // ex.set_light_mode(true);
    // ex.set_num_threads(4);

    float prob_threshold = 0.25f;
    float nms_threshold = 0.45f;

    Yolo model(prob_threshold, nms_threshold); 

    std::vector<Object> proposals;
    auto start_time = system_clock::now();
    ex.input("images", in_pad);
    {
        ncnn::Mat out1;
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;
        int stride1 = 8;
        std::vector<Object> object8;
        ex.extract("output", out1);
        model.generate_proposals(anchors, in_pad, out1, stride1, object8);
        // std::cout<<object8.size()<<std::endl;  
        proposals.insert(proposals.end(), object8.begin(), object8.end());
    }
    {
        ncnn::Mat out2;
        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;
        int stride2 = 16; 
        std::vector<Object> object16;
        ex.extract("264", out2);
        model.generate_proposals(anchors, in_pad, out2, stride2, object16);
        proposals.insert(proposals.end(), object16.begin(), object16.end());
    }
    {
        ncnn::Mat out3;
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;
        int stride3=32;
        std::vector<Object> object32;
        ex.extract("267", out3);
        model.generate_proposals(anchors, in_pad, out3, stride3, object32);
        proposals.insert(proposals.end(), object32.begin(), object32.end());
    }
    model.qsort_descent_inplace(proposals);

    std::vector<int> picked;
    // std::cout<<proposals.size()<<std::endl;  
    model.nms_sorted_bboxes(proposals, picked);
    duration<double> diff = system_clock::now()-start_time;
    cout<<"time consuming:"<<diff.count()*1000<<"ms"<<endl;
    std::vector<Object> objects;
    int count = picked.size();
    // std::cout<<count<<std::endl;
    objects.resize(count);
    for(int i=0; i<count; i++)
    {
        objects[i]=proposals[picked[i]];

        float x0 = (objects[i].x - pad_w) / scale;
        float y0 = (objects[i].y - pad_h) / scale;
        float x1 = (objects[i].x + objects[i].w - pad_w) / scale;
        float y1 = (objects[i].y + objects[i].h - pad_h) / scale;

        x0 = max(min(x0, (float)(width-1)), 0.f);
        y0 = max(min(y0, (float)(height-1)), 0.f);
        x1 = max(min(x1, (float)(width-1)), 0.f);
        y1 = max(min(y1, (float)(height-1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1-x0;
        objects[i].h = y1-y0; 
    }

    for(Object& obj:objects)
    {
        cv::rectangle(
            src_img,
            cv::Rect(obj.x, obj.y, obj.w, obj.h),
            cv::Scalar(255,0,0),
            3
        );
        string label = format("%.2f", obj.prob);
        label = classnames_list[obj.label]+":"+label;
        putText(src_img, label, Point(obj.x, obj.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
    }
    float scale2 = 0.3f;
    cv::imshow("detect", src_img);
    cv::waitKey(0);
    return 0;
}