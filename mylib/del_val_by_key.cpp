//jin: 2016-04-13 16:16:06 delete keys specified by a txt

#include "my_lmdb.hpp"

int del_val_by_key(int argc, char** argv)
{
	if (argc != 3)
	{
		printf("Usage: /path/of/del_val_by_key db_fullname keylist\n");
		return -1;
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

int main(int argc, char** argv) 
{
	int ret = del_val_by_key(argc, argv);
	
	return ret;

}
