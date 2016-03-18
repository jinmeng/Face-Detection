// jin: this file is based on convert_imagenet.cpp  20160113 11:43
//  gflags::SetUsageMessage("Convert a set of negatives (candidates detected from a set images without faces) to the lmdb\n"
//        "format used as input for Caffe.\n"
//        "Usage:\n"
//        "    convert_NoFace_MySpace [FLAGS] LISTFILE ROOTFOLDER/ CANDIDATE_FOLDER/ MIN_PROB MAX_PROB DB_NAME DB_KEYLISTFILE\n"
//        "");
//
// basic steps:

// 1. read all image names into a vector
// 2. for each image, read all candidates into a vector with probability between MIN_PROB and MAX_PROB (say 0.5-0.6) from its candidate file 
// 3. Open new lmdb
// 4. convert all data in this vector into lmdb      
// 5. save img_cand_pairs into a list
// TODO: shuffle the list of img_can_pairs, then combine two lmdb into one by a predefined order(if the pos/neg in lmdb1 is 2:1, then read it 3 times and read lmdb2 1 time)

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

/*
vector<CNN_RECT> NMS(vector<CNN_RECT> &cnn_rect, float iou_thr=0.5)
{
	vector<bool> mask(cnn_rect.size(), true);
	vector<CNN_RECT> new_rect_vec;

	sort(cnn_rect.begin(), cnn_rect.end(), comp_rect);
	for(int i=0; i<cnn_rect.size(); i++)
	{
		

	}
	return new_rect_vec;
}
*/

//============================== utility functions ==============================
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

int load_candidates(const string &cand_root_folder, const string &img_name, const double min_prob, const double max_prob, 
    vector<vector<CNN_RECT> > &cand_all_img, size_t max_cand_per_img) 
{
    //string root_folder = cand_root_folder, img = img_name;
    string cand_file = cand_root_folder + img_name + ".cand";
    
    ifstream infile(cand_file.c_str());
    if(!infile)
    {
        cerr << "Failed to open file " << cand_file << endl;
        return -1;
    }
    
    //push back all candidates with probability between min_prob and max_prob
    size_t cnt = 0, total;
    vector<CNN_RECT> cand_per_img;
    
    infile >> total;
    
    for(size_t i=0; i<total; ++i)
    {
        float tmp;
        CNN_RECT cand;
        
        
        infile >> cand.x >> cand.y >> cand.width >> cand.height >> tmp >> cand.prob;
        if(cand.prob >= min_prob && cand.prob < max_prob){
            cand_per_img.push_back(cand);
            ++cnt;
        }
        
        if(cnt == max_cand_per_img)
            break;
            
        //jin:test
        //cout << img_name << "  " << cand.x << " " << cand.y << " " << cand.width << " " << cand.height <<  endl;		
		// apply NMS to cand_per_img    
    }    
    //cout << endl;
    
    cand_all_img.push_back(cand_per_img);
    
    return 0;
}

//jin: modified from caffe/util/io.hpp:ReadImageToDatum 2016-01-13 16:17:56 
//by default, only support gray scale images
           
bool ReadImageRectToDatumArr(const string& img_filename, const int height, const int width, const vector<CNN_RECT> &cand_per_img, Datum *datum_per_img) 
{
    //jin:test
    //cout << "img_filename = " << img_filename << endl;
        
//    cv::Mat cv_img;
    int label = 0; // all negatives by default
    int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;   //int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat cv_img_origin = cv::imread(img_filename, cv_read_flag);
    if (!cv_img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << img_filename;
        return false;
    }

//    if (height > 0 && width > 0) {
//        cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
//    } else {
//        cv_img = cv_img_origin;
//    }

    int num_channels = 1; //int num_channels = (is_color ? 3 : 1);
    int cand_size = cand_per_img.size();
    
    for(int i=0; i<cand_size; ++i)
    {
        CNN_RECT rect = cand_per_img[i];
            //jin: test
        //cout << "rect.x, y, width, height: " << rect.x << ", " << rect.y << ", " << rect.width << ", "<< rect.height  << endl;
    
        cv::Mat img_rect; 
        cv::Mat roi = cv_img_origin(cv::Rect(rect.x, rect. y, rect.width, rect.height));
        
        if(rect.width != width && rect.height != height )
        {   
            cv::resize(roi, img_rect, cv::Size(height, width));
        }
                  
        Datum datum;
        datum.set_channels(num_channels);
        datum.set_height(img_rect.rows);
        datum.set_width(img_rect.cols);
        datum.set_label(label);
        datum.clear_data();
        datum.clear_float_data();
        string* datum_string = datum.mutable_data();
    
        for (int h = 0; h < img_rect.rows; ++h) {
            for (int w = 0; w < img_rect.cols; ++w) {
                datum_string->push_back(
                static_cast<char>(img_rect.at<uchar>(h, w)));
            }
        }
        
        datum_per_img[i] = datum;
    }

    return true;
}

//jin: based on BING_WU\BING_test\CnnFace.cpp:nonMaxSup 2016-01-17 17:58:02 
//void NMS(const vector<

//============================== MAIN function =================================
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Convert a set of negatives (candidates detected from a set of images without faces) to lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_NoFace_MySpace [FLAGS] LISTFILE ROOTFOLDER/ CANDIDATE_FOLDER/ MIN_PROB MAX_PROB DB_NAME DB_KEYLISTFILE\n"
        "    convert_NoFace_MySpace [FLAGS] LISTFILE ROOTFOLDER/ CANDIDATE_FOLDER/ MIN_PROB MAX_PROB DB_NAME DB_KEYLISTFILE MAX_CAND_NUM\n"
        );

    //jin
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    //FLAGS_log_dir="./log";
    
    if (argc != 8 && argc != 9) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_NoFace_MySpace ");
        return 1;
    }

//=========================1. read all image names into a vector
 //   bool is_color = !FLAGS_gray; //jin: treat all images as gray
    std::ifstream infile(argv[1]);
    vector<string> img_names;
        
    if(load_strlist(infile, img_names)) { return -1; }

    LOG(INFO) << "Loaded a total of " << img_names.size() << " images." << endl;
   
    //test 
    //cout << "A total of " << img_names.size() << " images." << endl;

//=========================2. for each image, read all candidates into a vector with probability between MIN_PROB and MAX_PROB from its candidate file  
    string root_folder(argv[2]);
    string cand_root_folder(argv[3]);
    float min_prob = atof(argv[4]);
    float max_prob = atof(argv[5]);
    
    //store candidates for each image with probability between min_prob and max_prob
    //vector<pair<string, CNN_RECT> > img_cand_pairs;
    //vector<int> num_cand_per_img;
    vector<vector<CNN_RECT> > cand_all_img;
    size_t total_num = img_names.size();
    int cnt=0;
    size_t max_cand_per_img = (argc == 9)? atoi(argv[8]): 50; //set maximal number of candidates per image
    
    cout << "Start loading candidates... " << endl;
    for(size_t i=0; i<total_num; ++i)
    {
        if(load_candidates(cand_root_folder, img_names[i], min_prob, max_prob, cand_all_img, max_cand_per_img))
        {
            return -1;
        }
		cnt += cand_all_img[i].size();
        
       if( (i+1)  % 1000 == 0)
		{
            cout << i+1  << " ";				
		}
    }
    cout << "\nA total of " << cnt << " candidates with probability in [" << min_prob << ", " << max_prob << ")."  << endl;

//=========================3. Open new lmdb
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;
    
    // Open db
    const string &db_backend = FLAGS_backend;
    const char* db_path = argv[6];
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
    
        
    for (size_t i = 0; i < total_num; ++i) {
        const vector<CNN_RECT> &cand_per_img = cand_all_img[i];
        size_t cand_size = cand_per_img.size();
        Datum *datum_per_img = new Datum[cand_size];
        

        if (!ReadImageRectToDatumArr(root_folder + img_names[i] + ".jpg", resize_height, resize_width,
            cand_all_img[i], datum_per_img)) 
        {
            cerr << "ReadImageRectToDatumVec failed when reading " << img_names[i] << endl;
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
                
                if (count % 5000 == 0)  LOG(ERROR) << "Produced " << count << " candidates.";
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
    std::ofstream outfile(argv[7]);
    if(!outfile)
    {
        cerr << "Failed to open file " << argv[7] << endl;
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
