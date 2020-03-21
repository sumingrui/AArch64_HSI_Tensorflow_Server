// test the tf model

#include <string>
#include "tf.h"
#include "share.h"

int main()
{
    std::string sendImgPath;
    TF_2dcnn("/home/armin/repos/hsi_tf/tf_forward/dataset/","newrawSinglefile20190711140909", sendImgPath);
    log(info, "Result Image path: " + sendImgPath);
    return 0;
}