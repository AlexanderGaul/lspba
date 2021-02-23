#ifndef CAMERA_H
#define CAMERA_H

#include <ceres/ceres.h>


template <typename Scalar>
class RadialCamera {
  public:
    static constexpr int N = 6;
    typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
    typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
    typedef Eigen::Matrix<Scalar, N, 1> VecN;
    
    RadialCamera(const VecN& p) : param(p) {};
    
    /* TODO is it allowed to write over same varaible??? */
    /* maybe check this in the basalt headers?? */
    Vec2 project(const Vec3& p) const {
        Vec2 res;
        
        res[0] = p[0] / p[2];
        res[1] = p[1] / p[2];

        Scalar r = res.norm();
        Scalar distortion = 1. + r * r * (param[4] + r * r * (param[5]));
        res[0] = distortion * res[0];
        res[1] = distortion * res[1];

        res[0] = param[0] * res[0] + param[2];
        res[1] = param[1] * res[1] + param[3];
        
        return res;
    }
    
    Vec3 unproject(const Vec2& p) const {
        Vec3 res;
        Vec2 uncalibrated; /* TODO: rename */
        uncalibrated[0] = 1. / param[0] * (p[0] - param[2]);
        uncalibrated[1] = 1. / param[1] * (p[1] - param[3]);
        
        Scalar b1 = - param[4];
        Scalar b2 = 3. * param[4] * param[4] - param[5];
        Scalar b3 = - 12. * param[4] * param[4] * param[4] + 
                    8. * param[4] * param[5];
        /*
        Scalar b4 = 55. * param[4] * param[4] * param[4] * param[4] - 
                    55. * param[4] * param[4] * param[5] + 
                    5. * param[5] * param[5];
        Scalar b5 = -273. * param[4] * param[4] * param[4] * param[4] * param[4] + 
                    364. * param[4] * param[4] * param[4] * param[4] - 
                    78. * param[4] * param[5] * param[5];
        Scalar b6 = 1428. * param[4] * param[4] * param[4] * param[4] * param[4] * param[4] - 
                    2380. * param[4] * param[4] * param[4] * param[4] * param[5] + 
                    840. * param[4] * param[4] * param[5] * param[5] - 
                    35. * param[5] * param[5];
                */
        Scalar r = uncalibrated.norm();
        Scalar r2 = r * r;
        Scalar r4 = r2 * r2;
        Scalar r6 = r4 * r2;
        Scalar r8 = r6 * r2;
        Scalar r10 = r8 * r2;
        Scalar r12 = r10 * r2;
        
        Scalar Q = 1. + b1 * r2 + b2 * r4 + b3 * r6; // + b4 * r8 + b5 * r10 + b6 * r12;
        
        res[0] = uncalibrated[0] * Q;
        res[1] = uncalibrated[1] * Q;
        res[2] = Scalar(1.);
        return res;
    }
    
    
    
  private:
    VecN param;
};


#endif // CAMERA_H
