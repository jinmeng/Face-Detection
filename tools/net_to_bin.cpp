// This program takes in a trained network and an input blob, and then dumps
// all the intermediate blobs produced by the net to individual binary
// files stored in protobuffer binary formats.
// Usage:
//    dump_network input_net_param trained_net_param
//        input_blob output_prefix 0/1
// if input_net_param is 'none', we will directly load the network from
// trained_net_param. If the last argv is 1, we will do a forward-backward pass
// before dumping everyting, and also dump the who network.

#include <string>
#include <vector>

#include "fcntl.h"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"

#include "caffe/util/upgrade_proto.hpp"
#include <cstdio>
#include <cstdlib>

using namespace caffe;  // NOLINT(build/namespaces)


int write_blob(FILE* fp, Blob<float>* blob) {
  if (blob->count() <= 0) {
    return -1;
  }
  int buf[4];
  buf[0] = blob->num();
  buf[1] = blob->channels();
  buf[2] = blob->height();
  buf[3] = blob->width();
  //std::cout << std::endl << "write blob: " << buf[0] << " " << buf[1] << " " << buf[2] << " " << buf[3] << "\n";
  fwrite(buf, sizeof(int), 4, fp);
  const float* pdata = blob->cpu_data();
  fwrite(pdata, sizeof(float), blob->count(), fp);
  return 0;
}

int write_layer(FILE* fp, Layer<float>* layer) {

  const LayerParameter& layer_param = layer->layer_param();
  int ksize = 0;
  int stride = 0;
  int num_output = 0;
  int param_byte_size = 0;
  int slice_dim = 0;
  int operation = 0;
  int pad = 0;

  int i = 0;

  const vector<shared_ptr<Blob<float> > >& param_blobs =  layer->blobs();

  switch(layer->type()) {
  case LayerParameter_LayerType_CONVOLUTION:
    ksize = layer_param.convolution_param().kernel_size();
    stride = layer_param.convolution_param().stride();
    pad = layer_param.convolution_param().pad();
    num_output = layer_param.convolution_param().num_output();

    std::cout << "    kernel size: " << ksize << "\n";
    std::cout << "    stride: " << stride << "\n";
    std::cout << "    num_output: " << num_output << "\n";
    std::cout << "    pad: " << pad << "\n";

    param_byte_size = 4 * sizeof(int);
    for (i = 0; i < param_blobs.size(); ++i) {
      param_byte_size += 4 * sizeof(int) + param_blobs[i]->count() * sizeof(float); // 4 dims + float data
      std::cout << "      param blob " << i << " count: " << param_blobs[i]->count() << "\n";
    }
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    fwrite(&ksize, sizeof(int), 1, fp);
    fwrite(&stride, sizeof(int), 1, fp);
    fwrite(&num_output, sizeof(int), 1, fp);
    fwrite(&pad, sizeof(int), 1, fp);

    std::cout << "    all params byte size: " << param_byte_size << std::endl;
    for (i = 0; i < param_blobs.size(); ++i) {
      write_blob(fp, &(*(param_blobs[i])));
    }

    break;
  case LayerParameter_LayerType_POOLING:
    ksize = layer_param.pooling_param().kernel_size();
    stride = layer_param.pooling_param().stride();

    std::cout << "    kernel size: " << ksize << "\n";
    std::cout << "    stride: " << stride << "\n";

    param_byte_size = 2 * sizeof(int);
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    fwrite(&ksize, sizeof(int), 1, fp);
    fwrite(&stride, sizeof(int), 1, fp);

    break;
  case LayerParameter_LayerType_INNER_PRODUCT:

    num_output = layer_param.inner_product_param().num_output();
    std::cout << "    num_output: " << num_output << "\n";

    param_byte_size = 1 * sizeof(int);
    for (i = 0; i < param_blobs.size(); ++i) {
      param_byte_size += 4 * sizeof(int) + param_blobs[i]->count() * sizeof(float); // 4 dims + float data
      std::cout << "      param blob " << i << " count: " << param_blobs[i]->count() << "\n";
    }
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    fwrite(&num_output, sizeof(int), 1, fp);
    std::cout << "    all params byte size: " << param_byte_size << std::endl;
    for (i = 0; i < param_blobs.size(); ++i) {
      write_blob(fp, &(*(param_blobs[i])));
    }

    break;
  case LayerParameter_LayerType_RELU:

    param_byte_size = 0;
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    break;
  case LayerParameter_LayerType_SOFTMAX:
    param_byte_size = 0;
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    break;
  case LayerParameter_LayerType_SPLIT:
    param_byte_size = 0;
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    break;
  case LayerParameter_LayerType_CONCAT:
    param_byte_size = 0;
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    break;
  case LayerParameter_LayerType_FLATTEN:
    param_byte_size = 0;
    fwrite(&param_byte_size, sizeof(int), 1, fp);

    break;
  case LayerParameter_LayerType_SLICE:

    slice_dim = layer_param.slice_param().slice_dim();
    std::cout << "    slice_dim: " << slice_dim << std::endl;
    param_byte_size = 4;
    fwrite(&param_byte_size, sizeof(int), 1, fp);
    fwrite(&slice_dim, sizeof(int), 1, fp);
    break;

  case LayerParameter_LayerType_ELTWISE:
    operation = layer_param.eltwise_param().operation();
    std::cout << "    operation: " << operation;
    if (operation == 0) {
      std::cout << " (PROD)\n";
    } else if (operation == 1) {
      std::cout << " (SUM)\n";
    } else if (operation == 2) {
      std::cout << " (MAX)\n";
    } else {
      std::cout << "\n[ERROR] operation not known: " << std::endl;
    }

    param_byte_size = 4;
    fwrite(&param_byte_size, sizeof(int), 1, fp);
    fwrite(&operation, sizeof(int), 1, fp);
    break;
  default:
    param_byte_size = 0;
    fwrite(&param_byte_size, sizeof(int), 1, fp);
    std::cout << "[ERROR] Layer Type: " << layer->type_name() << " not supported.\n";
    return -1;
    break;
  }
  return 0;
}


int main(int argc, char** argv) {

  if (argc < 4) {
    printf("  Usage: net_to_bin net_prototxt net_trained_model save_bin_file\n");
    return 0;
  }

  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  shared_ptr<Net<float> > caffe_net;
  NetParameter net_param;
  string net_param_file = argv[1];
  ReadNetParamsFromTextFileOrDie(net_param_file, &net_param);

  std::cout << net_param.input_size() << std::endl;
  std::cout << net_param.input_dim_size() << std::endl;
  for (int i = 0; i < 4; ++i) {
    std::cout << net_param.input_dim(i) << std::endl;
  }

  caffe_net.reset(new Net<float>(argv[1]));
  caffe_net->CopyTrainedLayersFrom(argv[2]);

  FILE* fp = fopen(argv[3], "wb");
  for (int i = 1; i < 4; ++i) {
    int input_dim = net_param.input_dim(i);
    fwrite(&input_dim, sizeof(int), 1, fp);
  }


  const vector<string>& blob_names = caffe_net->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();
  int num_blobs = blobs.size();
  fwrite(&num_blobs, sizeof(int), 1, fp);


  for (int i = 0; i < num_blobs; ++i) {
    std::cout << "blob " << i << " name: " << blob_names[i] << std::endl;
  }

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net->layers();

  const vector<vector<int> >& net_bottom_id_vecs = caffe_net->bottom_id_vecs();
  const vector<vector<int> >& net_top_id_vecs =  caffe_net->top_id_vecs();

  int num_layers = layers.size();
  fwrite(&num_layers, sizeof(int), 1, fp);

  for (int i = 0; i < layers.size(); ++i) {
    Layer<float>* layer_ptr = &(*layers[i]);
    std::cout << "layer " << i << ":\n";
    std::cout << "  type: " << layer_ptr->type() << "\n";
    std::cout << "  type_name: " << layer_ptr->type_name() << "\n";

    int layer_type = layer_ptr->type();
    fwrite(&layer_type, sizeof(int), 1, fp);

    std::cout << "  bottom: ";
    int bottom_size = net_bottom_id_vecs[i].size();
    fwrite(&bottom_size, sizeof(int), 1, fp);
    fwrite(&net_bottom_id_vecs[i][0], sizeof(int), bottom_size, fp);

    for (int j = 0; j < net_bottom_id_vecs[i].size(); ++j) {
      std::cout << net_bottom_id_vecs[i][j] << " ";
    }
    std::cout << "\n  top: ";

    int top_size = net_top_id_vecs[i].size();
    fwrite(&top_size, sizeof(int), 1, fp);
    fwrite(&net_top_id_vecs[i][0], sizeof(int), top_size, fp);

    for (int j = 0; j < net_top_id_vecs[i].size(); ++j) {
      std::cout << net_top_id_vecs[i][j] << " ";
    }
    std::cout << std::endl;

    write_layer(fp, layer_ptr);


    std::cout << "\n------\n";
  }

  fclose(fp);


/*

  vector<Blob<float>* > input_vec;
  shared_ptr<Blob<float> > input_blob(new Blob<float>());
  if (strcmp(argv[3], "none") != 0) {
    BlobProto input_blob_proto;
    ReadProtoFromBinaryFile(argv[3], &input_blob_proto);
    input_blob->FromProto(input_blob_proto);
    input_vec.push_back(input_blob.get());
  }

  string output_prefix(argv[4]);
  // Run the network without training.
  LOG(ERROR) << "Performing Forward";
  caffe_net->Forward(input_vec);
  if (argc > 5 && strcmp(argv[5], "1") == 0) {
    LOG(ERROR) << "Performing Backward";
    Caffe::set_phase(Caffe::TRAIN);
    caffe_net->Backward();
    // Dump the network
    NetParameter output_net_param;
    caffe_net->ToProto(&output_net_param, true);
    WriteProtoToBinaryFile(output_net_param,
        output_prefix + output_net_param.name());
  }
  // Now, let's dump all the layers

  const vector<string>& blob_names = caffe_net->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();
  for (int blobid = 0; blobid < caffe_net->blobs().size(); ++blobid) {
    // Serialize blob
    LOG(ERROR) << "Dumping " << blob_names[blobid];
    BlobProto output_blob_proto;
    blobs[blobid]->ToProto(&output_blob_proto);
    WriteProtoToBinaryFile(output_blob_proto,
        output_prefix + blob_names[blobid]);
  }
*/


  return 0;
}
