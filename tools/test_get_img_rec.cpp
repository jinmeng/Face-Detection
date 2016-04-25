//jin 2016 0329 12:53 check the output of get_img_rect
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::cout;
using std::cerr;
using std::endl;
using std::ends;
using std::vector;
using std::pair;
using std::make_pair;
using std::string;
using std::ifstream;

DEFINE_bool(gray, true,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

#define PRINT 
// struct 
typedef struct cnn_rectangle_{
    int x;
    int y;
    int width;
    int height;
	float prob; //used for NMS
}CNN_RECT;

bool comp_rect(const CNN_RECT &rect1, const CNN_RECT &rect2)
{
	return rect1.prob < rect2.prob;
}

//============================== utility functions ==============================
int bilinear_image_resize(const unsigned char *src, int width, int height,
								unsigned char *dst, int nw, int nh)
{
	double dx, dy, x, y;
	int x1, y1, x2, y2;	
	int i, j;	
	
	unsigned char *pout = dst;
	double f1, f2, f3, f4, f12, f34, val;
	
	
	//输入图像的采样间隔
	dx = (double) (width) / (double) (nw);
	dy = (double) (height) / (double) (nh);
	
	for (j=0; j<nh; ++j) 
	{
		y = j * dy;
		y1 = (int) y;
		y2 = y1 + 1;
		y2 = y2>height-1? height-1 : y2;
		for (i=0; i<nw; ++i) 
		{
			x = i * dx;
			x1 = (int) x;
			x2 = x1 + 1;
			x2 = x2>width-1? width-1 : x2;
			
			f1 = *(src + width * y1 + x1);
			f3 = *(src + width * y1 + x2);
			f2 = *(src + width * y2 + x1);
			f4 = *(src + width * y2 + x2);
			
			f12 = (f1  + (y-y1) * (f2 - f1));
			f34 = (f3  + (y-y1) * (f4 - f3));
			
			//+0.5是为了四舍五入，加0.00001是为了尽量避开取整时的大误差情况
			val = f12 + (x - x1) * (f34 - f12) + 0.5;
			if(val > 255)
				*pout = 255;
			else
				*pout = (unsigned char) val;
			++pout;
		}
	}
	
	return 0;
}
								 

//jin: added at 2016-02-23 09:24:41 
void get_img_rect(unsigned char *image, int width,  int height,
			 const CNN_RECT &rect, 
			 unsigned char *img_rect)
{
	int x = rect.x, y = rect.y, ht = rect.height, wd = rect.width; 
	
	for(int ix = 0; ix < wd; ++ix)
	{
		for(int iy = 0; iy < ht; ++iy)
		{
			// 分情况:x<0，y<0，x+wd>width, y+ht > height
			if(ix+x < 0 || y+iy < 0 || y+iy >= height || x+ix >= width)
			{
				img_rect[iy*wd + ix] = 0;
			}
			else{
				img_rect[iy*wd + ix] = image[(y+iy)*width + ix +x];
			}
			
#ifdef PRINT
			printf("%d  ", img_rect[iy*wd + ix]);
#endif
		}
		
#ifdef PRINT
		printf("\n");
#endif
	}
}
				 
//============================== MAIN function =================================

int main() {

	const string img_filename = "/home/jin/face_detection/caffe-rc/0003.jpg";
	
    cv::Mat cv_img_origin = cv::imread(img_filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!cv_img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << img_filename;
        return false;
    }
    
    //jin: 2016-02-23 09:18:42 convert mat into unsigned char and call get_img_rect instead of using cv::Rect
    unsigned char *image = cv_img_origin.data; 
    int width = cv_img_origin.cols, height = cv_img_origin.rows;
    int num_channels = 1; //int num_channels = (is_color ? 3 : 1);
	CNN_RECT rect = {1012, 210, 12, 12};

    unsigned char* img_rect = (unsigned char*) malloc(rect.width*rect.width*sizeof(unsigned char));
    if(NULL == img_rect){
        printf("Failed to malloc.\n");
        return false;
    }   
    
	get_img_rect(image, width, height, rect, img_rect);

	free(img_rect);
    
  return 0;
}
