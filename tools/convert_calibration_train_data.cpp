// jin: this file is based on convert_imagenet.cpp  20160113 11:43
//
// jin: 2016-02-23 08:30:48 add code (vector<bool> img_mask(true);) to deal with the case that after loading img_names into a list, either an image or its candidate does not exist 
// jin: 2016-02-23 09:20:31 add get_img_rect function to replace cv::Rect
// basic steps:

// 1. read all image names into a vector
// 2. for each image, read all ground truths and labels into a vector from its gt file 
// 3. Open new lmdb
// 4. convert all data in this vector into lmdb      
// 5. save img_cand_pairs into a list

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <iomanip>
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
using std::string;
using std::ifstream;

DEFINE_bool(gray, true,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");


// struct 
typedef struct rect_label{
    int x;
    int y;
    int width;
    int height;
	int label;
}RECT_LABEL;


//============================== utility functions ==============================
//jin: added at 20160223 11:19 copied from project:NS_Face:ns_fd_cnn_proposal.cpp
int bilinear_image_resize(const unsigned char *src, int width,int height,
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

int  load_strlist(ifstream &infile, vector<string> &strlist)
{
   if(!infile)
    {
        cerr << "Failed to open ifstream " << endl;
        return -1;
    }

    string filename;
    int total_num;
    infile >> total_num;

    for(int i=0; i<total_num; ++i)
    {
        infile >> filename;
        strlist.push_back(filename);
        //if(i<20)
        //    cout << img_names[i] << endl;
    }

    return 0;    
}

//jin: 2016-02-18 11:48:17  now the input img_name has suffix, so use substr to deal with it.
//int load_rect_label(const string &rect_label_root_folder, const string &img_name, vector<vector<RECT_LABEL> > &rect_label_all_img) 
//{
//    string img_str=img_name.substr(0, img_name.rfind("."));
//    string cand_file = rect_label_root_folder + img_str + ".fr";
//jin2016-02-24 08:27:36 change the parameter list
int load_rect_label(const string &img_name, const char *cand_fmt, vector<vector<RECT_LABEL> > &rect_label_all_img) 
{
    string cand_file=img_name.substr(0, img_name.rfind(".")) + cand_fmt;
    //cout << cand_file << endl; 
       
    ifstream infile(cand_file.c_str());
    if(!infile)
    {
        cerr << "Failed to open file " << cand_file << endl;
        return -1;
    }
    
    //push back all candidates with probability between min_prob and max_prob
    size_t  total;
    vector<RECT_LABEL> rect_label_per_img;
    
    infile >> total;
    
    for(size_t i=0; i<total; ++i)
    {
        RECT_LABEL gt;
        
        infile >> gt.x >> gt.y >> gt.width >> gt.height >> gt.label;
        rect_label_per_img.push_back(gt);
        
        //jin:test
        //cout << img_name << "  " << cand.x << " " << cand.y << " " << cand.width << " " << cand.height <<  endl;		
		// apply NMS to cand_per_img    
    }    
    //cout << endl;
    
    rect_label_all_img.push_back(rect_label_per_img);
    
    return 0;
}

//jin: added at 2016-02-23 09:24:41 
void get_img_rect(unsigned char *image, int width,  int height,
			 const RECT_LABEL &rect, 
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
		}
	}
}
//jin: modified from caffe/util/io.hpp:ReadImageToDatum 2016-01-13 16:17:56 
//by default, only support gray scale images
           
bool ReadImageRectToDatumArr(const string& img_filename, const int resize_height, const int resize_width, const vector<RECT_LABEL> &cand_per_img, Datum *datum_per_img) 
{
    //jin:test
    //cout << "img_filename = " << img_filename << endl;
    if(resize_height <= 0 || resize_width <= 0 || resize_height != resize_width)
    {
        cerr<<"resize_height <=0 or resize_width <=0 or resize_height != resize_width" << endl;
        return false;
    }     
        
    int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;   
	//int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat cv_img_origin = cv::imread(img_filename, cv_read_flag);
//    if (!cv_img_origin.data) { //jin: 20160226 17:26 change to empty() according to yu shiqi's libfacedetect-example.cpp
    if (cv_img_origin.empty()) { 
        LOG(ERROR) << "Could not open or find file " << img_filename;
        return false;
    }


    //jin: 2016-02-23 09:18:42 convert mat into unsigned char and call get_img_rect instead of using cv::Rect
 //   unsigned char *image = cv_img_origin.data; //jin: 20160226 17:26 change to .ptr(0) according to yu shiqi's libfacedetect-example.cpp
    unsigned char *image = cv_img_origin.ptr(0); 
    int width = cv_img_origin.cols, height = cv_img_origin.rows;
    int num_channels = 1; //int num_channels = (is_color ? 3 : 1);
    int cand_size = cand_per_img.size();
    
    //jin: determine maximal size needed to malloc 
    int  max_width = 0;
    for(int i=0; i<cand_size; ++i)
    {
       if (max_width < cand_per_img[i].width)
       {    max_width = cand_per_img[i].width; }
    }

    unsigned char* img_rect = (unsigned char*) malloc(max_width*max_width*sizeof(unsigned char));
    if(NULL == img_rect){
        printf("Failed to malloc.\n");
        return false;
    }
    
    unsigned char* img_resize = (unsigned char*) malloc(resize_height*resize_width*sizeof(unsigned char));
    if(NULL == img_rect){
        printf("Failed to malloc.\n");
        return false;
    }
    
    for(int i=0; i<cand_size; ++i)
    {
        RECT_LABEL rect = cand_per_img[i];
        get_img_rect(image, width, height, rect, img_rect);
        if(rect.width != resize_width && rect.height != resize_height )
        {   
            bilinear_image_resize(img_rect, rect.width, rect.height, img_resize, resize_width, resize_height);  
        }
        else
        {   
            int rect_size = rect.width*rect.height;
            for(int k=0; k<rect_size; ++k)
            {
                img_resize[k] = img_rect[k];
            }
        }
        
        Datum datum;
        datum.set_channels(num_channels);
        datum.set_height(resize_height);
        datum.set_width(resize_width);
        datum.set_label(rect.label);
        datum.clear_data();
        datum.clear_float_data();
        string* datum_string = datum.mutable_data();
    
        for (int h = 0; h < resize_height; ++h) {
            for (int w = 0; w < resize_width; ++w) {
                datum_string->push_back(img_resize[h*resize_width+w]);
            }
        }
        
        datum_per_img[i] = datum;
    
    }
    
    free(img_rect);
    free(img_resize);
    
    return true;
}

//============================== MAIN function =================================
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of ground truths and labels into lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_calibration_train_data [FLAGS] LISTFILE ROOTFOLDER/ RECT_LABEL_FOLDER/ DB_NAME DB_KEYLISTFILE\n"
        );

    //jin
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_log_dir="./log";
    
    if (argc != 6) {
        cout << "argc = " << argc << endl;
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_calibration_train_data");
        return 1;
    }

//=========================1. read all image names into a vector
 //   bool is_color = !FLAGS_gray; //jin: treat all images as gray
    std::ifstream infile(argv[1]);
    vector<string> img_names;
    
        
    if(load_strlist(infile, img_names)) { return -1; }

    LOG(INFO) << "Loaded a list of " << img_names.size() << " image names." << endl;

//=========================2. for each image, read all candidates into a vector with probability between MIN_PROB and MAX_PROB from its candidate file  
    string img_root_folder(argv[2]);
    string rect_label_root_folder(argv[3]);
    
    vector<vector<RECT_LABEL> > rect_label_all_img;
    size_t total_num = img_names.size();
    vector<bool> img_mask(total_num, true);
    int cnt=0;
    
    cout << "Start loading ground truths(x, y, width, heigth) and labels... " << endl;
    for(size_t i=0; i<total_num; ++i)
    {
        string gt_name = rect_label_root_folder + img_names[i];
        
        if(load_rect_label(gt_name, ".fr", rect_label_all_img))
        {
             cout << "img_mask[" << i << "]=" << img_mask[i] << endl;
            img_mask[i] = false;                   
			continue; 
        }
		cnt += rect_label_all_img[i].size();
        
        if( (i+1)  % 500 == 0)
		{
            cout << std::setw(6) << i+1 << std::flush;				
		}
    }
    cout << "\nA total of " << cnt << " ground truths with labels."  << endl;
	if(0==cnt) return -1;
    
//=========================3. Open new lmdb
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;
    
    // Open db
    const string &db_backend = FLAGS_backend;
    const char* db_path = argv[4];
    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);
    
    if (db_backend == "lmdb") {  // lmdb
        LOG(INFO) << "Opening lmdb " << db_path;
        CHECK_EQ(mkdir(db_path, 0744), 0)
            << "mkdir " << db_path << "failed";
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
            << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
            << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";
    } else 
    {
        LOG(FATAL) << "Unknown db backend " << db_backend;
    }
  
//=========================4. convert all data in this vector into lmdb      // Storing to db
    //maximum number of items in a lmdb
    //int max_num = 310000; //1000100;  //test  3 images
    int count = 0; 
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];
    //to save all key string into file DB_KEYLISTFILE
    vector<string> keystr_vec;
    
    //jin: find all those jpgs that can't be opened in ReadImageRectToDatumArr()  2016-02-24 09:10:13 
    std::ofstream outjpg("wrong_jpg.txt");
    if(!outjpg)
    {
        cerr << "Failed to open file " << "wrong_jpg.txt" << endl;
        return -1;
    }

    
    for (size_t i = 0; i < total_num && true == img_mask[i]; ++i) {            
        const vector<RECT_LABEL> &cand_per_img = rect_label_all_img[i];
        size_t cand_size = cand_per_img.size();
        Datum *datum_per_img = new Datum[cand_size];
       	
		string imgstr = img_root_folder + img_names[i];
		if(!ReadImageRectToDatumArr(imgstr, resize_height, resize_width, rect_label_all_img[i], datum_per_img)) 
        {
            //cerr << "ReadImageRectToDatumVec failed when reading " << imgstr << endl;
            outjpg << imgstr << endl;
            continue;
        }	
        
        // sequential datums in image img_names[i]       
        //jin: test
        //cout << "image index i = " << i << ", cand_size = " << cand_size << endl;
        
        for(size_t j = 0; j < cand_size; ++j)
        {
            snprintf(key_cstr, kMaxKeyLength, "%08d_%s_%d_%d_%d_%d", count,
                img_names[i].c_str(), cand_per_img[j].x, cand_per_img[j].y, cand_per_img[j].width, cand_per_img[j].height);
            
            //jin:test
            //cout << "key_cstr = " << key_cstr << endl;
            
            string value;         
            datum_per_img[j].SerializeToString(&value);
            string keystr(key_cstr);
            keystr_vec.push_back(keystr);
            
            // Put in db       
            if (db_backend == "lmdb") {  // lmdb
                mdb_data.mv_size = value.size();
                mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
                mdb_key.mv_size = keystr.size();
                mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
                CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
                  << "mdb_put failed";
            } else {
                LOG(FATAL) << "Unknown db backend " << db_backend;
            }           
            if (++count % 1000 == 0) {
              // Commit txn
                if (db_backend == "lmdb") {  // lmdb
                    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_commit failed";
                    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_begin failed";
                } else {
                    LOG(FATAL) << "Unknown db backend " << db_backend;
                }             
                if (count % 5000 == 0)  LOG(INFO) << "Produced " << count << " candidates.";
            }           
        }
        
        // write the last batch
        if (total_num-1 == i && count % 1000 != 0) {
            if (db_backend == "lmdb") 
            {  // lmdb
                CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
                mdb_close(mdb_env, mdb_dbi);
                mdb_env_close(mdb_env);
            } else {
                LOG(FATAL) << "Unknown db backend " << db_backend;
            }
            LOG(ERROR) << "Produced " << count << " candidates.";
        } 
        
        delete [] datum_per_img;     
//        if(count >= max_num)           break;   
    }

//=========================5. save img_cand_pairs into a list
    std::ofstream outfile(argv[5]);
    if(!outfile)
    {
        cerr << "Failed to open file " << argv[5] << endl;
        return -1;
    }
    
    size_t keystr_vec_size = keystr_vec.size();
    outfile << keystr_vec_size << endl;
    for(size_t i = 0; i < keystr_vec_size; ++i)
    {
        outfile << keystr_vec[i] << endl;
    }
    
  return 0;
}
