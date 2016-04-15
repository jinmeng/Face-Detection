//jin: 20160411 12:28 rewrite code for combining two lmdbs, refer to combine_lmdb2.cpp

#include "my_lmdb.hpp"

int combine(int argc, char** argv)
{
	//0. check number of arguments
	if (argc != 8) {
		printf("Usage: \n/path/of/combine_lmdb [FLAGS] SRC_DB1 SRC_DB1_KEYLIST SRC_DB2 SRC_DB2_KEYLIST \nTAR_DB TAR_DB_KEYLIST ERR_KEYLIST\n");	
        return -1;
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

int main(int argc, char** argv) 
{
	//int ret = del_key(argc, argv);
	
	//TODO: there are too many invalid value whose size are 0
	int ret = combine(argc, argv);
	
	return ret;
}




