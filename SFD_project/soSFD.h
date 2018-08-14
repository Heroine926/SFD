#include <stdio.h>
#include<stdlib.h>
#include <vector>
using namespace std;
extern "C"
{    
    int sfd_face_detection(const void *p_cvmat, void *p_vec_vec_rect, int batchSize);

    int sfd_face_detection_init(const string& model_file, const string& trained_file,int gpu_id);
}
