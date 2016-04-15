//jin: create my own lib for the use of my_lmdb.cpp
#ifndef MY_LMDB_HPP
#define MY_LMDB_HPP

#include <iosfwd>
#include <memory>
#include <lmdb.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <sys/stat.h>

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::string;
using std::vector;
using std::pair;
using std::make_pair;

using std::endl;


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
	
		printf("datum.channels() = %d, datum.height() = %d, datum.width() = %d\n",datum_.channels(), datum_.height(), datum_.width());
			 
		const string& data = datum_.data();
		int data_size = data.size();
		printf("data.size() = %d\n", data_size); //cout << "data.size() = " << data_size << endl;
		
		if (data_size != 0) {		 
			for (int i = 0; i < data_size; ++i) {
				char tmp = ((i+1) % datum_.height() == 0) ? '\n':'\t';
				printf("%d%c", (uint8_t)data[i], tmp);
			}
		}
	}
	
	void Imshow(const string &str);
	
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

const size_t LMDB_MAP_SIZE = 1099511627776; // 1 TB

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
int load_keylist(vector<string> &strlist, const char *strlistname);

//int combine(int argc, char** argv);

//int del_val_by_key(int argc, char** argv);

#endif
