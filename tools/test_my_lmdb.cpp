//jin: 20160412 20:04 deprecated： use code in caffe-rc/mylib instead
//jin: 20160411 12:28 rewrite code for combining two lmdbs, refer to combine_lmdb2.cpp
//jin 20160405 16:53  try to wrap the lmdb interface with cpp and delete data specified by keys
//	input: lmdb, keystring
//	output: delete the data corresponding to the key
//jin: 20160324 13:23 refer to https://github.com/BVLC/caffe/blob/master/include/caffe/util/db.hpp
// 	and caffe-rc/include/util/data_layers.hpp

//print for test
//#define PRINT 

#include <iosfwd>
#include <memory>
#include <lmdb.h>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/scoped_ptr.hpp>

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

using boost::scoped_ptr;
//======================================================================class header
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
	classname(const classname&);\
	classname& operator=(const classname&)
   
enum Mode { READ, WRITE, NEW };

inline void MDB_CHECK(int mdb_status){
	CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class Cursor{
public:
	explicit Cursor(MDB_txn *mdb_txn, MDB_cursor *mdb_cursor)
		: mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false){
		SeekToFirst();
	}
	
	virtual ~Cursor(){
		mdb_cursor_close(mdb_cursor_);
		mdb_txn_abort(mdb_txn_);
	}
	
	void SeekToFirst(){ Seek(MDB_FIRST); }
	
	//after call SeekByKey, check if valid ==true, and then call value() 
	void SeekByKey(const string &str)
	{
		string keystr = str;
	    mdb_key_.mv_size = keystr.size();
	    mdb_key_.mv_data = reinterpret_cast<void*>(&keystr[0]);  // const_cast and static_cast NOT work
		Seek(MDB_SET_KEY);
	}
	
	void DelByKey(const string &keystr){
		SeekByKey(keystr);
		
		if(valid_ == true){
			//printf("deleted value for key %s\n", keystr.c_str());
			int rc = mdb_cursor_del(mdb_cursor_, 0);
			MDB_CHECK(rc);
		}
		else
			printf("keystr %s not found\n", keystr.c_str());
	}
	
	int Commit(){ return mdb_txn_commit(mdb_txn_);	}
	
	void Next(){ Seek(MDB_NEXT); }
	
	string Key() {
		return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
	}

	string Value() {
		return string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);
	}
	
	string Value(const string &keystr){
		SeekByKey(keystr);
		return Valid() ? Value() : "";
	}
	
	bool Valid() {  return valid_;	}
	
	//copied from print_lmdb_by_key.cpp
	void Print()
	{
		//Datum datum;
		datum_.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
		cout << "datum.channels() = " << datum_.channels() << ", datum.height() = " << datum_.height() 
			 << ", datum.width() = " << datum_.width() << endl;
		const string& data = datum_.data();
		int data_size = data.size();
		if (data_size != 0) {
			cout << "data.size() = " << data_size << endl;
			for (int i = 0; i < data_size; ++i) {
				char tmp = ((i+1) % datum_.height() == 0) ? '\n':'\t';
				printf("%d%c", (uint8_t)data[i], tmp);
			}
		}
		else
		    printf("data.size() == 0!");
	}
	
	void Imshow(const string &str)
	{
		printf("keystr = %s\n", str.c_str());
		datum_.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);	
		
		const string &data = datum_.data();
		
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
	void Seek(MDB_cursor_op op) 
	{
		int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_,
										&mdb_value_, op);
		if(mdb_status == MDB_NOTFOUND)
			valid_ = false;
		else{
			MDB_CHECK(mdb_status);
			valid_ = true;
		}
	}
	
	MDB_txn *mdb_txn_;
	MDB_cursor *mdb_cursor_;
	MDB_val mdb_key_, mdb_value_;
	bool valid_;
	Datum datum_;
};

class Transaction{
public:
	explicit Transaction(MDB_dbi *mdb_dbi, MDB_txn *mdb_txn):
		mdb_dbi_(mdb_dbi), mdb_txn_(mdb_txn){}
	
	virtual void Put(const string &key, const string &value)
	{
		MDB_val mdb_key, mdb_value;
		
		mdb_key.mv_data = const_cast<char*>(key.data());
		mdb_key.mv_size = key.size();
		
		mdb_value.mv_data = const_cast<char*>(value.data());
		mdb_value.mv_size = value.size();
		
		MDB_CHECK(mdb_put(mdb_txn_, *mdb_dbi_, &mdb_key, &mdb_value, 0));
	}
	
	virtual void Commit(){ MDB_CHECK(mdb_txn_commit(mdb_txn_));}
	
private:
	MDB_dbi *mdb_dbi_;
	MDB_txn *mdb_txn_;
	
	DISABLE_COPY_AND_ASSIGN(Transaction);
};

class LMDB
{
public:
	LMDB() : mdb_env_(NULL) { }
	
	virtual ~LMDB() { Close(); }
	
	void Open(const string &source, Mode mode);
	
	void Close() {
		if (mdb_env_ != NULL) {
			mdb_dbi_close(mdb_env_, mdb_dbi_);
			mdb_env_close(mdb_env_);
			mdb_env_ = NULL;
		}
	}
	
	virtual Cursor * NewCursor(unsigned int flags = MDB_RDONLY)
	{
		MDB_txn *mdb_txn;
		MDB_cursor *mdb_cursor;
		MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, flags, &mdb_txn));//jin change MDB_RDONLY to flags
		MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
		MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
		
		return new Cursor(mdb_txn, mdb_cursor);
	}
	
	virtual Transaction * NewTransaction()
	{
		MDB_txn *mdb_txn;
		MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
		MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
		
		return new Transaction(&mdb_dbi_, mdb_txn);
	}

private:
	MDB_env* mdb_env_;
	MDB_dbi mdb_dbi_;
	
	DISABLE_COPY_AND_ASSIGN(LMDB);
};

//======================================================================class source code
const size_t LMDB_MAP_SIZE = 1099511627776; // 1 TB

void LMDB::Open(const string &source, Mode mode)
{
	MDB_CHECK(mdb_env_create(&mdb_env_));
	MDB_CHECK(mdb_env_set_mapsize(mdb_env_, LMDB_MAP_SIZE));
	if (mode == NEW) {
		CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir" << source << "failed";
	}
	int flags = 0;
	if (mode == READ) {
		flags = MDB_RDONLY | MDB_NOTLS;
	}
	int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);
	
#ifndef ALLOW_LMDB_NOLOCK
	MDB_CHECK(rc);
#else
	if (rc == EACCES) {
		 cout << "Permission denied. Trying with MDB_NOLOCK ...";
		//LOG(WARNING) << "Permission denied. Trying with MDB_NOLOCK ...";
		// Close and re-open environment handle
		mdb_env_close(mdb_env_);
		MDB_CHECK(mdb_env_create(&mdb_env_));
		// Try again with MDB_NOLOCK
		flags |= MDB_NOLOCK;
		MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
	} else {
		MDB_CHECK(rc);
	}
#endif
	
	cout << "Opened lmdb " << source << endl; // LOG(INFO) << "Opened lmdb " << source;
}

//====================================================================== helper
//TODO  try to replace it with ReadDB
int load_keylist(vector<string> &strlist, const char *strlistname)
{ 
    std::ifstream infile(strlistname);
    if(!infile)
    {
        cerr << "Failed to open ifstream " << strlistname << endl;
        return -1;
    }

    string keystr;
	float tmp;
    int total_num;
    infile >> total_num;

    //for(int i=0; i<total_num; ++i)
	while(infile >> keystr >> tmp)
    {
        //infile >> keystr ;
        strlist.push_back(keystr);             
    }

    infile.close();
    return 0;    
}

class WriteDB  
{  
public:  
	WriteDB(){}
	
	WriteDB(const string &dbName):db_(dbName.c_str()){	}
	
	bool is_open()	{ return (db_.is_open())? true : false;	}
	
    bool Open(const string &dbName) {
		db_.open(dbName.c_str()); 
		return is_open();
	}  
	
	void Write(const int sz) {db_ << sz << endl;}
	
	void Write(const pair<string, float> &p) { db_ << p.first << "  " << p.second << endl;}
	
	void Write(const vector<pair<string, float> > &vp) 
	{
		int sz = vp.size();
		Write(sz);
		for(int i=0; i<sz; ++i)
		{
			Write(vp[i]);
		}
	}
	
	void Write(const vector<string> &sv) {
		int sz = sv.size();
		Write(sz);
		for(int i = 0; i < sz; ++i)
		{
			db_ << sv[i] << endl;  
		}
	} 
	
//	void write(const float &data) { db_<<data; }    
	
	virtual ~WriteDB() {  db_.close(); }  
	
private:	
    std::ofstream db_;  
}; 

class ReadDB  
{  
public:  
	ReadDB(){}
	
	ReadDB(const string &dbName):db_(dbName.c_str()){	}
	
	bool is_open()	{ return (db_.is_open())? true : false;	}
	
    bool Open(const string &dbName) {
		db_.open(dbName.c_str()); 
		return is_open();
	}  
	
	void Read(int &sz) { db_ >> sz; }
	
	void Read(pair<string, float> &p) { db_ >> p.first >> p.second;}
	
	void Read(vector<pair<string, float> > &vp) 
	{
		pair<string, float> p;
		
		while (!db_.eof() ) 
		{
			Read(p);
			vp.push_back(p);
		}
	}
	
	void Read(vector<string> &vs) {
		string s;
		
		while (!db_.eof() ) 
		{
			db_ >> s;  
			vs.push_back(s);
		}
	}    
	
	virtual ~ReadDB() {  db_.close(); }  
	
private:	
    std::ifstream db_;  
};

//====================================================================== sub functions

int del_key(int argc, char** argv)
{
	if (argc != 3)
	{
		printf("Usage:\n./my_lmdb db_fullname keylist");
	}
	
	string src_db(argv[1]);
	
	const char * keylist_txt = argv[2];
	vector<string> vk; 
	if(0 != load_keylist(vk, keylist_txt)){
		printf("load_keylist failed\n");
		return -1;
	}
	
	scoped_ptr<LMDB> lmdb(new LMDB());
	lmdb->Open(src_db, WRITE);
	
	//scoped_ptr<Transaction> txn(lmdb->NewTransaction());
	scoped_ptr<Cursor> cursor(lmdb->NewCursor(0));
	
	int i;
	for(i=0; i<vk.size(); i++)
	{
		cursor->DelByKey(vk[i]);
	}
	
	cursor->Commit();
	
	printf("Deleted %d pairs of key/value", i);
	
	return 0;
}

int combine(int argc, char** argv)
{
	//0. check number of arguments
	if (argc != 8) {
		printf("Usage: combine_lmdb2 [FLAGS] SRC_DB1 SRC_DB1_KEYLIST SRC_DB2 SRC_DB2_KEYLIST TAR_DB TAR_DB_KEYLIST ERR_KEYLIST\n"
			 );	
        return 1;
    }
	const char *src_db_path1 = argv[1], *keylist1 = argv[2], *src_db_path2 = argv[3], *keylist2 = argv[4]; 
	const char *tar_db_path = argv[5], *tar_db_keylist = argv[6], *err_db_keylist = argv[7];
	
	//1. read keystrings into vector for both input lmdb
	ReadDB db1(keylist1), db2(keylist2);
    if(!db1.is_open() || !db2.is_open()) {	
		return -1; 
	}
	
	int sz_db1;
	db1.Read(sz_db1);
	vector<string> kv_db1;
	db1.Read(kv_db1);
	//shuffle(kv_db1.begin(), kv_db1.end());
	
	int sz_db2;
	db2.Read(sz_db2);
	vector<string> kv_db2;
	db2.Read(kv_db2);
	//shuffle(kv_db2.begin(), kv_db2.end());
	
	kv_db1.reserve(sz_db1 + sz_db2);
	std::copy(kv_db2.begin(), kv_db2.end(), std::back_inserter(kv_db1));
	shuffle(kv_db1.begin(), kv_db1.end());
	
	//2. open input lmdbs and output lmdb
	scoped_ptr<LMDB> src_lmdb1(new LMDB());
	src_lmdb1->Open(src_db_path1, READ);
	scoped_ptr<Cursor> cursor1(src_lmdb1->NewCursor());
	
	scoped_ptr<LMDB> src_lmdb2(new LMDB());
	src_lmdb2->Open(src_db_path2, READ);
	scoped_ptr<Cursor> cursor2(src_lmdb2->NewCursor());
	
	scoped_ptr<LMDB> tar_lmdb(new LMDB());
	tar_lmdb->Open(tar_db_path, NEW);
	scoped_ptr<Transaction> txn(tar_lmdb->NewTransaction());
	
	//3. read key/value from src_lmdb1 and src_lmdb2 and write into tar_lmdb
	int count = 0;
    const int kMaxKeyLength = 256;
	
    vector<string> db_keylist3; //to save keylists for the combined lmdb
	vector<string> invalid_key; //save keystrings with datum of size 0;
    
    for( int i = 0; i < kv_db1.size(); ++i ) //jin: now the sizes of both lmdb1 and lmdb2 are 1000000  
    {	
        //find the value by the key in lmdb1           
        string keystr = kv_db1[i];
		string value;
		cursor1->SeekByKey(keystr);
		cursor2->SeekByKey(keystr);
     	
        //check if keystr1 is neither in lmdb1 nor in lmdb2
        if(!cursor1->Valid()  && !cursor2->Valid())  
        {
			cout << "key string " << keystr << " NOT found in either " << src_db_path1 << " or " << src_db_path2;
            return -1;
		}
		else if(cursor1->Valid()){
			//check if mdb_value1 is of size 0  //20160407
			if (0 == cursor1->Value().size()){
				invalid_key.push_back(keystr);
				//printf("value for %s is invalid!\n", keystr1.c_str());
				continue;
			}	
			value = cursor1->Value();	
			
#ifdef PRINT  //test
			cursor1->Imshow(keystr); //cursor1->Print();
#endif
        } 
        else  if (cursor2->Valid()) //no, keystr1 is a key in lmdb2
        {
			//check if mdb_value1 is of size 0  //20160407
			if (0 == cursor2->Value().size()){
				invalid_key.push_back(keystr);
				//printf("value for %s is invalid!\n", keystr1.c_str());
				continue;
			}	
			value = cursor2->Value();
#ifdef PRINT			//test
			cursor2->Imshow(keystr);//cursor2->Print();
#endif
        }

		//convert keystr1 in lmdb1/lmdb2 and put into lmdb3    
		char key_cstr[kMaxKeyLength];  
		snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count, keystr.c_str());		
		string keystr3(key_cstr);
		db_keylist3.push_back(keystr3);
		
		txn->Put(keystr3, value);
		
		//LOG(INFO) << keystr3;
		//print_datum_data(mdb_value2);//jin: print data in datum. refer to compute_image_mean.cpp::123
		
        if (++count % 1000 == 0) { 
			txn->Commit();
			txn.reset(tar_lmdb->NewTransaction());
            //CHECK_EQ(mdb_txn_commit(mdb_txn3), MDB_SUCCESS)  << "mdb_txn_commit failed";
            //CHECK_EQ(mdb_txn_begin(mdb_env3, NULL, 0, &mdb_txn3), MDB_SUCCESS)  << "mdb_txn_begin failed";

            if (count % 10000 == 0) 
				printf("Processed %d items\n", count);
        }
    }
	
	if (count % 1000 != 0) {  
		txn->Commit();
		printf("Processed %d items\n", count);
	}
  
	
	//4. write keystrings of tar_lmdb 
	WriteDB db3(tar_db_keylist), db4(err_db_keylist);
	if(!db3.is_open() || !db4.is_open()) {	
		return -1; 
	}

	db3.Write(db_keylist3);
	
	db4.Write(invalid_key);
	
	return 0;
}

//====================================================================== main function
int main(int argc, char** argv) 
{
	//int ret = del_key(argc, argv);
	int ret = combine(argc, argv);
	
	return ret;
}




