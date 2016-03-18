// jin: Load key string from input lmdb 2016-01-15 12:39:20 
//  gflags::SetUsageMessage("Load key string from input lmdb\n"
//        "format used as input for Caffe.\n"
//        "Usage:\n"
//        "    load_keystr_from_lmdb [FLAGS] DB_NAME DB_KEYLISTFILE\n"
//        "");
//
// basic steps:

// 1. Open lmdb 
// 2. read all key strings from lmdb, save into a vector
// 3. save vector into a list


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

//DEFINE_bool(gray, true,    "When this option is on, treat images as grayscale ones");
//DEFINE_bool(shuffle, false,    "Randomly shuffle the order of images and their labels");
//DEFINE_string(backend, "lmdb", "The backend for storing the result");
//DEFINE_int32(resize_width, 0, "Width images are resized to");
//DEFINE_int32(resize_height, 0, "Height images are resized to");


//============================== MAIN function =================================
int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Load key string from input lmdb\n"
        "Usage:\n"
        "    load_keystring_from_lmdb [FLAGS] db_name db_keylist\n"
        "");
    
    //jin
    //gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_log_dir="./log";
    
    if (argc != 3) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/load_keystring_from_lmdb ");
        return 1;
    }

//========================= 1. Open lmdb 
    const char *src_db1 = argv[1];
    
    MDB_env *mdb_env1; 
    MDB_dbi mdb_dbi1;  
    MDB_val mdb_key1, mdb_value1;  
    MDB_txn *mdb_txn1;  
    MDB_cursor *mdb_cursor1;  
   
    CHECK_EQ(mdb_env_create(&mdb_env1), MDB_SUCCESS) << "mdb_env_create failed";          
    CHECK_EQ(mdb_env_set_mapsize(mdb_env1, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";        
    CHECK_EQ(mdb_env_open(mdb_env1, src_db1, MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";       
    CHECK_EQ(mdb_txn_begin(mdb_env1, NULL, MDB_RDONLY, &mdb_txn1), MDB_SUCCESS) << "mdb_txn_begin failed";      
    CHECK_EQ(mdb_open(mdb_txn1, NULL, 0, &mdb_dbi1), MDB_SUCCESS) << "mdb_open failed";        
    LOG(INFO) << "Opening lmdb in " << src_db1 << endl;        
          
	CHECK_EQ(mdb_cursor_open(mdb_txn1, mdb_dbi1, &mdb_cursor1), MDB_SUCCESS)
			<< "mdb_cursor_open failed";
	CHECK_EQ(mdb_cursor_get(mdb_cursor1, &mdb_key1, &mdb_value1, MDB_FIRST), MDB_SUCCESS) << "mdb_cursor_get failed";

    
//========================= 2. read all key strings from lmdb, save into a vector
    //to save all key string into file DB_KEYLISTFILE
    vector<string> keystr_vec;
    
    int count = 0;     
    //const int kMaxKeyLength = 256;
    //char key_cstr[kMaxKeyLength];
	//memcpy(key_cstr, mdb_key1.mv_data, mdb_key1.mv_size);

    //string keystr(key_cstr, mdb_key1.mv_size);
    string keystr((char *)mdb_key1.mv_data, mdb_key1.mv_size);
    keystr_vec.push_back(keystr);
    
//	cout << keystr << endl;
    count++;
   
    while(mdb_cursor_get(mdb_cursor1, &mdb_key1, &mdb_value1, MDB_NEXT) == MDB_SUCCESS)
    {
        //char key_cstr[kMaxKeyLength];
        //memcpy(key_cstr, mdb_key1.mv_data, mdb_key1.mv_size);
        //string keystr(key_cstr, mdb_key1.mv_size);
        string keystr((char *)mdb_key1.mv_data, mdb_key1.mv_size);
        
        keystr_vec.push_back(keystr);
        count++;
        
        if(count % 10000 == 0)
        {
            cout << count << "  " << ends;
        }
                
    }
    
    cout << endl << src_db1 << " contains count = " << count << " key/value pairs. "<< endl;
//========================= 3. save vector into a list
    std::ofstream outfile(argv[2]); 
    if(!outfile)
    {
        cerr << "Failed to open file " << argv[7] << endl;
        return -1;
    }
    
    int keystr_vec_size = keystr_vec.size();
    outfile << keystr_vec_size << endl;
    
    for(int i = 0; i < keystr_vec_size; ++i)
    {
        outfile << keystr_vec[i] << endl;
    } 
    
    
 
  return 0;
}
