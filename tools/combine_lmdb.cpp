// jin: this file is based on caffe-rc/src/caffe/layers/data_layer.cpp  2016-01-14 10:09:43 
//      finished at 2016-01-15 11:32:21 

//  gflags::SetUsageMessage("Combine two lmdbs into one\n"
//        "Usage:\n"
//        "    combine_lmdb [FLAGS] SRC_DB1 SRC_DB1_KEYLIST SRC_DB2 SRC_DB2_KEYLIST TAR_DB TAR_DB_KEYLIST\n"
//        "");
//
// basic steps:

// 1.1 read key strings from SRC_DB1_LIST into vector db_keystrlist1; 
// 1.2 read key strings from SRC_DB2_LIST into vector dbvec2, shuffle it
// 2. read pairs of keys and values from original lmdb1 (both positives and negatives) and a second lmdb2 (containing all negatives) and write into lmdb3
// 2.1 open lmdb1 for reading values from lmdb1 by key strings which are converted from key strings in db_keylist1
// 2.2 open lmdb2 for reading values by keys 
// 2.3 open lmdb3 
// 2.4 write key/value pairs read from lmdb1 and lmdb2 into lmdb3, since #lmdb1 = 1500000 and #lmdb2 = 50000, we read 3 times from lmdb1 and 1 time from lmdb2     
// 3. save keystrins of lmdb3 into a file of list 

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
using std::vector;
using std::pair;
using std::make_pair;
using std::string;
using std::ifstream;

DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(max_pos, 0, "maximal size of positives");
DEFINE_int32(max_neg, 0, "maximal size of negatives");

//============================== utility functions =================================
int load_strlist(ifstream &infile, vector<string> &strlist)
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
        //test
        
    }

    return 0;    
}

void print_datum_data(const MDB_val &mdb_value1)
{
    Datum datum;
    datum.ParseFromArray(mdb_value1.mv_data, mdb_value1.mv_size);
	cout << "datum.channels() = " << datum.channels() << ", datum.height() = " << datum.height() << ", datum.width() = " << datum.width() << endl;
    const string& data = datum.data();
    int data_size = data.size();
    if (data_size != 0) {
       cout << "data_size = " << data_size << endl;
       for (int i = 0; i < data_size; ++i) {
           char tmp = ((i+1)%datum.height() == 0) ? '\n':'\t';
           printf("%d%c", (uint8_t)data[i], tmp);
       }
    }
}

//============================== MAIN function =================================================
int main(int argc, char** argv) 
{
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    gflags::SetUsageMessage("Combine two lmdbs into one.\n"
        "Usage:\n"
        "    combine_lmdb [FLAGS] SRC_DB1 SRC_DB1_KEYLIST SRC_DB2 SRC_DB2_KEYLIST TAR_DB TAR_DB_KEYLIST\n"
        "");
    
    //jin
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_log_dir="./log";

    if (argc != 7) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/combine_lmdb ");
        return 1;
    }
    
    const char *src_db_path1 = argv[1], *src_db_path2 = argv[3];  
    
// 1. read key strings from SRC_DB1_LIST into vector db_keystrlist1; read key strings from SRC_DB2_LIST into vector dbvec2, shuffle it
    std::ifstream infile1(argv[2]); 
    vector<string> db_keylist1;
    if(load_strlist(infile1, db_keylist1)) { return -1; }
   
    std::ifstream infile2(argv[4]);  
    vector<string> db_keylist2;
    if(load_strlist(infile2, db_keylist2)) { return -1; }
    
    //data in lmdb2 are stored in the order that all rectangles in the same images are near each other, therefore, it should be shuffled
    shuffle(db_keylist2.begin(), db_keylist2.end());
     
    int max_neg = FLAGS_max_neg;
    
    if(max_neg > 0 && max_neg < db_keylist2.size())
    {
        db_keylist2.resize(max_neg);
    }
     
// 2. read pairs of keys and values from original lmdb1 (both positives and negatives) and a second lmdb2 (containing all negatives) and write into lmdb3
// 2.1 open lmdb1 for reading values by keys 
    MDB_env *mdb_env1; 
    MDB_dbi mdb_dbi1;  
    MDB_val mdb_key1, mdb_value1;  
    MDB_txn *mdb_txn1;  
//    MDB_cursor *mdb_cursor1;  
   
    CHECK_EQ(mdb_env_create(&mdb_env1), MDB_SUCCESS) << "mdb_env_create failed";          
    CHECK_EQ(mdb_env_set_mapsize(mdb_env1, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";        
    CHECK_EQ(mdb_env_open(mdb_env1, src_db_path1, MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";       
    CHECK_EQ(mdb_txn_begin(mdb_env1, NULL, MDB_RDONLY, &mdb_txn1), MDB_SUCCESS) << "mdb_txn_begin failed";      
    CHECK_EQ(mdb_open(mdb_txn1, NULL, 0, &mdb_dbi1), MDB_SUCCESS) << "mdb_open failed";        
    LOG(INFO) << "Opening source lmdb1 in " << src_db_path1 << endl;        
          
//	CHECK_EQ(mdb_cursor_open(mdb_txn1, mdb_dbi1, &mdb_cursor1), MDB_SUCCESS) << "mdb_cursor_open failed";
//	CHECK_EQ(mdb_cursor_get(mdb_cursor1, &mdb_key1, &mdb_value1, MDB_FIRST), MDB_SUCCESS) << "mdb_cursor_get failed";
	
//2.2 open lmdb2 for reading values by keys  
    MDB_env *mdb_env2; 
    MDB_dbi mdb_dbi2;  
    MDB_val mdb_key2, mdb_value2;  
    MDB_txn *mdb_txn2;  
//    MDB_cursor *mdb_cursor2; 
        
    CHECK_EQ(mdb_env_create(&mdb_env2), MDB_SUCCESS) << "mdb_env_create failed";           
    CHECK_EQ(mdb_env_set_mapsize(mdb_env2, 1099511627776), MDB_SUCCESS)   // 1TB           
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env2, src_db_path2, MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";       
    CHECK_EQ(mdb_txn_begin(mdb_env2, NULL, MDB_RDONLY, &mdb_txn2), MDB_SUCCESS)  << "mdb_txn_begin failed";    
    CHECK_EQ(mdb_open(mdb_txn2, NULL, 0, &mdb_dbi2), MDB_SUCCESS) << "mdb_open failed";        
    LOG(INFO) << "Opening source lmdb2 in " << src_db_path2; 
          
//2.3 open lmdb3 
    const char *tar_db = argv[5];
    MDB_env *mdb_env3;
    MDB_dbi mdb_dbi3;
    MDB_val mdb_key3, mdb_key4;
    MDB_txn *mdb_txn3;
          
    CHECK_EQ(mkdir(tar_db, 0744), 0) << "mkdir " << tar_db << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env3), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env3, 1099511627776), MDB_SUCCESS)  << "mdb_env_set_mapsize failed";// 1TB       
    CHECK_EQ(mdb_env_open(mdb_env3, tar_db, 0, 0664), MDB_SUCCESS)  << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env3, NULL, 0, &mdb_txn3), MDB_SUCCESS)   << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn3, NULL, 0, &mdb_dbi3), MDB_SUCCESS)  << "mdb_open failed. Does the lmdb already exist? ";
    LOG(INFO) << "Opening target lmdb in " << tar_db;
    
//2.4  write key/value pairs read from lmdb1 and lmdb2 into lmdb3, since #lmdb1 = 1500000 and #lmdb2 = 50000, we read 3 times from lmdb1 and 1 time from lmdb2     
    int i = 0, j = 0, count = 0;
    const int kMaxKeyLength = 256;
	string keystr3;
	
    vector<string> db_keylist3;//to save keylists for the combined lmdb
    
    while(i < db_keylist1.size() ) //jin: now the sizes of both lmdb1 and lmdb2 are 1000000  
    {
        //jin test
        //if(i==3)         return 0;
    
        //find the value by the key in lmdb1           
        string keystr1 = db_keylist1[i];
	    mdb_key1.mv_size = keystr1.size();
	    mdb_key1.mv_data = reinterpret_cast<void*>(&keystr1[0]);      

	    CHECK_EQ(mdb_get(mdb_txn1, mdb_dbi1, &mdb_key1, &mdb_value1), MDB_SUCCESS) << "mdb_get failed";                      
	    ++i;					       

	    //convert keys in lmdb1 into keys in lmdb3 and put  into lmdb3     
	    char key_cstr[kMaxKeyLength]; 
        snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count, keystr1.c_str());
        ++count;
        keystr3 = key_cstr;
        db_keylist3.push_back(keystr3);
        
        mdb_key3.mv_size = keystr3.size();
        mdb_key3.mv_data = reinterpret_cast<void*>(&keystr3[0]);
        CHECK_EQ(mdb_put(mdb_txn3, mdb_dbi3, &mdb_key3, &mdb_value1, 0), MDB_SUCCESS) << "mdb_put failed";
              
        //jin: test: print data in datum. This part of code is from compute_image_mean.cpp::123
        //LOG(INFO) << keystr3;
        //print_datum_data(mdb_value1);
                	
        //find the value by the key in lmdb2          
        //if(i%3 == 0)  //NOTE: uncomment and change 3 when lmdb1.size() != lmdb2.size()
        {
			string keystr2 = db_keylist2[j];
			mdb_key2.mv_size = keystr2.size();
			mdb_key2.mv_data = reinterpret_cast<void*>(&keystr2[0]);      		

			CHECK_EQ(mdb_get(mdb_txn2, mdb_dbi2, &mdb_key2, &mdb_value2), MDB_SUCCESS) << "mdb_put failed";                      
			++j;

			//convert keys in lmdb2 into keys in lmdb3 and put  into lmdb3    
			char key_cstr[kMaxKeyLength];  
			snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count, keystr2.c_str());
			++count;
			keystr3 = key_cstr;
			db_keylist3.push_back(keystr3);
        
			mdb_key4.mv_size = keystr3.size();
			mdb_key4.mv_data = reinterpret_cast<void*>(&keystr3[0]);
			CHECK_EQ(mdb_put(mdb_txn3, mdb_dbi3, &mdb_key4, &mdb_value2, 0), MDB_SUCCESS)  << "mdb_put failed";
			
			//jin: test
			//LOG(INFO) << keystr3;
			//print_datum_data(mdb_value2);
		}
                      
        if (count % 1000 == 0) {         
            CHECK_EQ(mdb_txn_commit(mdb_txn3), MDB_SUCCESS)  << "mdb_txn_commit failed";
            CHECK_EQ(mdb_txn_begin(mdb_env3, NULL, 0, &mdb_txn3), MDB_SUCCESS)  << "mdb_txn_begin failed";

            LOG(ERROR) << "Processed " << count << " files.";
        }
        
    }
    //LOG(INFO) << "i = " << i << ", j = " << j << endl;
    // write the last batch
    if (count % 1000 != 0) {                      
        CHECK_EQ(mdb_txn_commit(mdb_txn3), MDB_SUCCESS) << "mdb_txn_commit failed";
        
        mdb_close(mdb_env3, mdb_dbi3);
        mdb_env_close(mdb_env3);
      
        mdb_close(mdb_env2, mdb_dbi2);
        mdb_env_close(mdb_env2);
        
        mdb_close(mdb_env1, mdb_dbi1);
        mdb_env_close(mdb_env1);
        
        LOG(ERROR) << "Processed " << count << " files.";
    }           


// 3. save keystrins of lmdb3 into a file of list   
    std::ofstream outfile(argv[6]);
    if(!outfile)
    {
        cerr << "Failed to open file " << argv[6] << endl;
        return -1;
    }
    
    size_t keylist_vec_size = db_keylist3.size();
    outfile << keylist_vec_size << endl;
    for(size_t i = 0; i < keylist_vec_size; ++i)
    {
        outfile << db_keylist3[i] << endl;
    } 
    
                
  return 0;
}
