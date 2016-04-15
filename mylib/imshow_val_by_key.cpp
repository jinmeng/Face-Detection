//jin: 2016-04-13 16:16:06 delete keys specified by a txt

#include "my_lmdb.hpp"

int imshow_val_by_key(int argc, char** argv)
{
	if (argc != 3)
	{
		printf("Usage: /path/of/imshow_val_by_key /path/of/lmdb keystring\n");
		return -1;
	}
	
	string src_db(argv[1]);
	
	const char * keystr = argv[2];
	
	scoped_ptr<LMDB> lmdb(new LMDB());
	lmdb->Open(src_db, READ);
	
	scoped_ptr<Cursor> cursor(lmdb->NewCursor());
	
	cursor->Imshow(keystr);

	return 0;
}

int main(int argc, char** argv) 
{
	int ret = imshow_val_by_key(argc, argv);
	
	return ret;

}
