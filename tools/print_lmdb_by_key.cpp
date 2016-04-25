//jin 20160331 17:50
//input: lmdb, keystring
//output: print the data corresponding to the key

#include <iosfwd>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  

using namespace cv; 

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using std::vector;
using std::pair;
using std::make_pair;

using std::cout;
using std::endl;
using std::cerr;


//jin: 20160324 13:23 refer to https://github.com/BVLC/caffe/blob/master/include/caffe/util/db.hpp
// and caffe-rc/include/util/data_layers.hpp
enum Mode { READ, WRITE, NEW };
class LMDB
{
public:
	LMDB() : mdb_env_(NULL) { }
	
	virtual ~LMDB() { close(); }
	
	void open(const char *source)
	{
		CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";          
		CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS)  // 1TB
				<< "mdb_env_set_mapsize failed";        
		CHECK_EQ(mdb_env_open(mdb_env_, source, MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";       
		CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "mdb_txn_begin failed";      
		CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "mdb_open failed";        
		LOG(INFO) << "Opening source lmdb1 in " << source << endl;                  
		CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "mdb_cursor_open failed";
	}
	
	void close() {
		if (mdb_env_ != NULL) {
			mdb_dbi_close(mdb_env_, mdb_dbi_);
			mdb_env_close(mdb_env_);
			mdb_env_ = NULL;
		}
	}
	
	void set_key(const string &str){ 
		string keystr = str;
		mdb_key_.mv_size = keystr.size();
		mdb_key_.mv_data = reinterpret_cast<void*>(&keystr[0]); 
	}
	
	const string& datum(const string &str)
	{
		set_key(str);
		CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_SET_KEY), MDB_SUCCESS) << "mdb_cursor_get failed";
		datum_.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);	
		
		return datum_.data();
	}
	
	//string value(const string &str) {
	//	set_key(str);
	//	CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_SET_KEY), MDB_SUCCESS) << "mdb_cursor_get failed";
	//	return string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);
	//}	
	
	void print_value(const string &str)
	{
		printf("keystr = %s\n", str.c_str());
		
		const string &data = datum(str);
		
		
		int data_size = data.size();
		
		printf("data.size() = %d\n", data_size);
		
		for(int i=0; i < data.size(); ++i)
		{
			char tmp = ((i+1) % datum_.height() == 0) ? '\n':'\t';
			
			printf("%d%c", (uint8_t)data[i], tmp);
		}	
			
	}
	
	void imshow(const string &str)
	{
		printf("keystr = %s\n", str.c_str());

		const string &data = datum(str);
		
		int ht = datum_.height(), wd = datum_.width();
		cv::Mat img(ht, wd, CV_8UC1);
		
		for(int i=0; i<ht; i++){
			for(int j=0; j<wd; j++){
				img.at<uint8_t>(i,j) = (uint8_t)data[i*ht+j];
			}
		}
		
		cv::namedWindow( "Image", 1 );//创建窗口
        cv::imshow( "Image", img );//显示图像
		
        cv::waitKey(0); //等待按键
		
        cv::destroyWindow( "Image" );//销毁窗口
        //cv::releaseImage( &img ); //释放图像
	}

	
private:
	// LMDB
	MDB_env* mdb_env_;
	MDB_dbi mdb_dbi_;
	MDB_txn* mdb_txn_;
	MDB_cursor* mdb_cursor_;
	MDB_val mdb_key_, mdb_value_;
	
	Datum datum_;
};



int main(int argc, char** argv) 
{
	const char *src_db_path1 = argv[1]; //argv[3];
	
	string keystr = argv[2];
	
	LMDB lmdb;
	lmdb.open(src_db_path1);
	
	lmdb.print_value(keystr);

	lmdb.imshow(keystr);
}




