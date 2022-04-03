#include <ctime>
#include <string>
#include <fstream>
#include <cstdlib>
#include <iostream>

#include <eigen3/Eigen/Eigen>

#include <pcl/io/pcd_io.h>
#include <pcl/common/pca.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>

#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/surface/on_nurbs/triangulation.h>
#include <pcl/surface/on_nurbs/fitting_surface_tdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>

using namespace pcl;
using namespace std;
clock_t Start,End;
bool isInterface(pcl::PointXYZ _point,pcl::PointCloud<pcl::PointXYZ>::Ptr corner_4)
{
    Eigen::Vector3f v1,v2,v3,v4;
    float d1 = pcl::geometry::squaredDistance(corner_4->points[0],corner_4->points[1]);
    float d2 = pcl::geometry::squaredDistance(corner_4->points[0],corner_4->points[2]);
    float d3 = pcl::geometry::squaredDistance(corner_4->points[0],corner_4->points[3]);

    if (d1>d2 && d1>d3)
    {
        v2.x()=corner_4->points[0].x;
        v2.y()=corner_4->points[0].y;
        v2.z()=corner_4->points[0].z;
        
        v3.x()=corner_4->points[1].x;
        v3.y()=corner_4->points[1].y;
        v3.z()=corner_4->points[1].z;

        v1.x()=corner_4->points[2].x;
        v1.y()=corner_4->points[2].y;
        v1.z()=corner_4->points[2].z;

        v4.x()=corner_4->points[3].x;
        v4.y()=corner_4->points[3].y;
        v4.z()=corner_4->points[3].z;
    }
    else if (d2>d1 && d2>d3)
    {
        v2.x()=corner_4->points[0].x;
        v2.y()=corner_4->points[0].y;
        v2.z()=corner_4->points[0].z;
        
        v3.x()=corner_4->points[2].x;
        v3.y()=corner_4->points[2].y;
        v3.z()=corner_4->points[2].z;

        v1.x()=corner_4->points[1].x;
        v1.y()=corner_4->points[1].y;
        v1.z()=corner_4->points[1].z;

        v4.x()=corner_4->points[3].x;
        v4.y()=corner_4->points[3].y;
        v4.z()=corner_4->points[3].z;
    }
    else if (d3>d1 && d3>d2)
    {
        v2.x()=corner_4->points[0].x;
        v2.y()=corner_4->points[0].y;
        v2.z()=corner_4->points[0].z;
        
        v3.x()=corner_4->points[3].x;
        v3.y()=corner_4->points[3].y;
        v3.z()=corner_4->points[3].z;

        v1.x()=corner_4->points[1].x;
        v1.y()=corner_4->points[1].y;
        v1.z()=corner_4->points[1].z;

        v4.x()=corner_4->points[2].x;
        v4.y()=corner_4->points[2].y;
        v4.z()=corner_4->points[2].z;
    }
    //首先判断是否在半个三角形内
    Eigen::Vector3f cross_res = (v1-v2).cross(v1-v3);
    double d = -(v1.cwiseProduct(cross_res).x()+v1.cwiseProduct(cross_res).y()+v1.cwiseProduct(cross_res).z());
    double A=cross_res.x();
    double B=cross_res.y();
    double C=cross_res.z();
    double X=(B*B*_point.x+C*C*_point.x-A*B*_point.y-A*C*_point.z-A*d)/(A*A+B*B+C*C);
    double Y=B/A*(X-_point.x)+_point.y;
    double Z=C/A*(X-_point.x)+_point.z;
    Eigen::Vector3f project_point;
    project_point.x()=X;
    project_point.y()=Y;
    project_point.z()=Z;
    Eigen::Vector3f vec_tmp1=(project_point-v1).cross(v2-v1);
    Eigen::Vector3f vec_tmp2=(project_point-v2).cross(v3-v2);
    Eigen::Vector3f vec_tmp3=(project_point-v3).cross(v1-v3);
    if (vec_tmp1.dot(vec_tmp2)>0 && vec_tmp1.dot(vec_tmp3)>0 && vec_tmp2.dot(vec_tmp3)>0)
        return true;
    //继续判断是否在另一个三角形内
    cross_res = (v4-v2).cross(v4-v3);
    d = -(v4.cwiseProduct(cross_res).x()+v4.cwiseProduct(cross_res).y()+v4.cwiseProduct(cross_res).z());
    A=cross_res.x();
    B=cross_res.y();
    C=cross_res.z();
    X=(B*B*_point.x+C*C*_point.x-A*B*_point.y-A*C*_point.z-A*d)/(A*A+B*B+C*C);
    Y=B/A*(X-_point.x)+_point.y;
    Z=C/A*(X-_point.x)+_point.z;

    project_point.x()=X;
    project_point.y()=Y;
    project_point.z()=Z;
    vec_tmp1=(project_point-v2).cross(v3-v2);
    vec_tmp2=(project_point-v3).cross(v4-v3);
    vec_tmp3=(project_point-v4).cross(v2-v4);
    if (vec_tmp1.dot(vec_tmp2)>0 && vec_tmp1.dot(vec_tmp3)>0 && vec_tmp2.dot(vec_tmp3)>0)
        return true;
    return false;
    /* Eigen::Vector3f res;
    res.x()=_point.x;
    res.y()=_point.y;
    res.z()=_point.z;
    Eigen::Vector3f vec_tmp1=(res-v1).cross(v2-v1);
    Eigen::Vector3f vec_tmp2=(res-v2).cross(v4-v2);
    Eigen::Vector3f vec_tmp3=(res-v4).cross(v3-v4);
    Eigen::Vector3f vec_tmp4=(res-v3).cross(v1-v3);
    if (vec_tmp1.dot(vec_tmp2)>0 && vec_tmp1.dot(vec_tmp3)>0 && vec_tmp1.dot(vec_tmp4)>0)
        return true;
    return false; */
}

void
PointCloud2Vector3d (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::on_nurbs::vector_vec3d &data);
void
visualizeCurve (ON_NurbsCurve &curve, ON_NurbsSurface &surface, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

double calculate_err_point(pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_origin,pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_triangle)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(_cloud_triangle);
  double result=0;
  for (auto points: _cloud_origin->points)
  {
    int K=1;//寻找最近的三个点
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    kdtree.nearestKSearch(points,K,pointIdxNKNSearch,pointNKNSquaredDistance);
    Eigen::Vector3f v;
    v.x() = _cloud_triangle->points[pointIdxNKNSearch[0]].x;
    v.y() = _cloud_triangle->points[pointIdxNKNSearch[0]].y;
    v.z() = _cloud_triangle->points[pointIdxNKNSearch[0]].z;
    Eigen::Vector3f this_points;
    this_points.x()=points.x;
    this_points.y()=points.y;
    this_points.z()=points.z;
    result += (v-this_points).norm();
  }
  result = result / _cloud_origin->points.size();
  return result;
}

int main(int argc , char *argv[])
{
    //===============================================================
    //加载点云
    //pcl::PointCloud<PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
    //if (pcl::io::loadPCDFile("cloud_projected.pcd",*cloud_projected)<0)
    //{
    //    CalculateProjectedCloud();
    //}
	pcl::PointCloud<PointXYZ>::Ptr cloud_origin (new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile("20220301涡轮叶片_叶背_L1.pcd", *cloud_origin)<0)
    {
        PCL_ERROR("\\origin点云文件不存在!\\\n");
        system("pause");
        return -1;
    }
    cout<<"->加载点的个数:"<<cloud_origin->points.size()<<endl;
    //================================================================
    //创建可视化类viewer指针
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("viewer"));
    //viewer->setSize (1000,600);//指定viewer大小
    //================================================================
    //将点云数据以重心为原点，重构点云数据
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_origin,centroid);
    pcl::PointCloud<PointXYZ>::Ptr cloud_recon (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_recon->points.resize(cloud_origin->points.size());
    cloud_recon->header = cloud_origin->header;
    cloud_recon->width = cloud_origin->width;
    cloud_recon->height = cloud_origin->height;
    for (size_t point_i = 0;point_i<cloud_origin->points.size();point_i++)
    {
        cloud_recon->points[point_i].x=cloud_origin->points[point_i].x-centroid[0];
        cloud_recon->points[point_i].y=cloud_origin->points[point_i].y-centroid[1];
        cloud_recon->points[point_i].z=cloud_origin->points[point_i].z-centroid[2];
    }
    pcl::io::savePCDFile("cloud_recon.pcd",*cloud_recon);

    //================================================================
    //计算PCA
    //ofstream outfile("test.txt");
    pcl::PCA<PointXYZ> pca;
    pca.setInputCloud(cloud_recon);
    int row_start = 0 , col_start = 0;
    size_t nb_rows = cloud_origin->height;
    size_t nb_cols = cloud_origin->width;
    pca.setIndices(row_start,col_start,nb_rows,nb_cols);//设置感兴趣点索引
    cout << "->特征值（从大到小）:\n"<<pca.getEigenValues()<<endl;
    cout << "->特征值对应的特征向量:\n"<<pca.getEigenVectors()<<endl;
    //===============================================================
    //建立投影平面
    Eigen::Matrix3f EigenVectors = pca.getEigenVectors();
    float a = EigenVectors(0,2);
    float b = EigenVectors(1,2);
    float c = EigenVectors(2,2);
    //float d = -(a*centroid[0]+b*centroid[1]+c*centroid[2]);
    float d = 0;
    //===============================================================
    //创建project_inliers投影模型
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
    coefficients->values.resize(4);
    coefficients->values[0] = a;
    coefficients->values[1] = b;
    coefficients->values[2] = c;
    coefficients->values[3] = d;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud_recon);
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);
    //pcl::io::savePCDFile("cloud_projected.pcd",*cloud_projected);
    //===============================================================
    //以原点为中心网格点建立网格坐标    
    float step = 1000;//设置步长
    std::vector<pcl::PointXYZ> grid_point;
    pcl::PointXYZ grid;
    grid.x = 0 - step*EigenVectors(0,0);
    grid.y = 0 - step*EigenVectors(0,1);
    grid.z = 0 - step*EigenVectors(0,2);
    grid_point.push_back(grid);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;//建立kdtree搜索
    kdtree.setInputCloud(cloud_projected);
    int K = 1;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    std::vector<int> point_index;
    int i = 0,j = 0,k = 0;
    std::vector<pcl::PointXYZ> grid_point_2;
    std::vector<pcl::PointXYZ> grid_point_3;
    //std::cout<<grid_point[0]<<std::endl;
    //=============================================================
    //主程序循环，寻找网格点
    while (1)
    {
        grid.x = grid_point[i].x + step*EigenVectors(0,0);
        grid.y = grid_point[i].y + step*EigenVectors(0,1);
        grid.z = grid_point[i].z + step*EigenVectors(0,2);
        kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
        //进行kdtree搜索最近点
        if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
        {
           break;
        }
        //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
        grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
        grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
        grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
        point_index.push_back(pointIdxNKNSearch[0]);
        grid_point_2.push_back(grid);
        grid_point_3.push_back(grid);
        grid_point.push_back(grid);
        
        j=0;
        while(1)
        {
            grid.x = grid_point_2[j].x + step*EigenVectors(1,0);
            grid.y = grid_point_2[j].y + step*EigenVectors(1,1);
            grid.z = grid_point_2[j].z + step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index.push_back(pointIdxNKNSearch[0]);
            grid_point_2.push_back(grid);

            j++;
        }
        grid_point_2.clear();
        k=0;
        while(1)
        {
            grid.x = grid_point_3[k].x - step*EigenVectors(1,0);
            grid.y = grid_point_3[k].y - step*EigenVectors(1,1);
            grid.z = grid_point_3[k].z - step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index.push_back(pointIdxNKNSearch[0]);
            grid_point_3.push_back(grid);

            k++;
        }
        grid_point_3.clear();
        i++;
    }
    i = 0;
    grid_point.clear();
    grid.x = 0;
    grid.y = 0;
    grid.z = 0;
    grid_point.push_back(grid);
    while (1)
    {
        grid.x = grid_point[i].x - step*EigenVectors(0,0);
        grid.y = grid_point[i].y - step*EigenVectors(0,1);
        grid.z = grid_point[i].z - step*EigenVectors(0,2);
        kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
        //进行kdtree搜索最近点
        if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
        {
           break;
        }
        //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
        grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
        grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
        grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
        point_index.push_back(pointIdxNKNSearch[0]);
        grid_point.push_back(grid);
        grid_point_2.push_back(grid);
        grid_point_3.push_back(grid);
        j=0;
        while(1)
        {
            grid.x = grid_point_2[j].x + step*EigenVectors(1,0);
            grid.y = grid_point_2[j].y + step*EigenVectors(1,1);
            grid.z = grid_point_2[j].z + step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index.push_back(pointIdxNKNSearch[0]);
            grid_point_2.push_back(grid);

            j++;
        }
        grid_point_2.clear();
        k=0;
        while(1)
        {
            grid.x = grid_point_3[k].x - step*EigenVectors(1,0);
            grid.y = grid_point_3[k].y - step*EigenVectors(1,1);
            grid.z = grid_point_3[k].z - step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index.push_back(pointIdxNKNSearch[0]);
            grid_point_3.push_back(grid);

            k++;
        }
        grid_point_3.clear();
        i++;
    } 
    grid_point.clear();
    std::cout<<"grid point detected!"<<std::endl;
    //================================================================
    //寻找面点
    grid.x = - 1.5*step*EigenVectors(0,0)-0.5*step*EigenVectors(1,0);
    grid.y = - 1.5*step*EigenVectors(0,1)-0.5*step*EigenVectors(1,1);
    grid.z = - 1.5*step*EigenVectors(0,2)-0.5*step*EigenVectors(1,2);
    std::vector<int> point_index_2;
    grid_point.push_back(grid);
    i=0;
    while (1)
    {
        grid.x = grid_point[i].x + step*EigenVectors(0,0);
        grid.y = grid_point[i].y + step*EigenVectors(0,1);
        grid.z = grid_point[i].z + step*EigenVectors(0,2);
        kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
        //进行kdtree搜索最近点
        if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
        {
           break;
        }
        //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
        grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
        grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
        grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
        point_index_2.push_back(pointIdxNKNSearch[0]);
        grid_point_2.push_back(grid);
        grid_point_3.push_back(grid);
        grid_point.push_back(grid);
        
        j=0;
        while(1)
        {
            grid.x = grid_point_2[j].x + step*EigenVectors(1,0);
            grid.y = grid_point_2[j].y + step*EigenVectors(1,1);
            grid.z = grid_point_2[j].z + step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index_2.push_back(pointIdxNKNSearch[0]);
            grid_point_2.push_back(grid);

            j++;
        }
        grid_point_2.clear();
        k=0;
        while(1)
        {
            grid.x = grid_point_3[k].x - step*EigenVectors(1,0);
            grid.y = grid_point_3[k].y - step*EigenVectors(1,1);
            grid.z = grid_point_3[k].z - step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index_2.push_back(pointIdxNKNSearch[0]);
            grid_point_3.push_back(grid);

            k++;
        }
        grid_point_3.clear();
        i++;
    }
    i = 0;
    grid_point.clear();
    grid.x =  -0.5*step*EigenVectors(0,0)+0.5*step*EigenVectors(1,0);
    grid.y =  -0.5*step*EigenVectors(0,1)+0.5*step*EigenVectors(1,1);
    grid.z =  -0.5*step*EigenVectors(0,2)+0.5*step*EigenVectors(1,2);
    grid_point.push_back(grid);
    while (1)
    {
        grid.x = grid_point[i].x - step*EigenVectors(0,0);
        grid.y = grid_point[i].y - step*EigenVectors(0,1);
        grid.z = grid_point[i].z - step*EigenVectors(0,2);
        kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
        //进行kdtree搜索最近点
        if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
        {
           break;
        }
        //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
        grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
        grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
        grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
        point_index_2.push_back(pointIdxNKNSearch[0]);
        grid_point.push_back(grid);
        grid_point_2.push_back(grid);
        grid_point_3.push_back(grid);
        j=0;
        while(1)
        {
            grid.x = grid_point_2[j].x + step*EigenVectors(1,0);
            grid.y = grid_point_2[j].y + step*EigenVectors(1,1);
            grid.z = grid_point_2[j].z + step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index_2.push_back(pointIdxNKNSearch[0]);
            grid_point_2.push_back(grid);

            j++;
        }
        grid_point_2.clear();
        k=0;
        while(1)
        {
            grid.x = grid_point_3[k].x - step*EigenVectors(1,0);
            grid.y = grid_point_3[k].y - step*EigenVectors(1,1);
            grid.z = grid_point_3[k].z - step*EigenVectors(1,2);
            kdtree.nearestKSearch (grid,K,pointIdxNKNSearch,pointNKNSquaredDistance);
            if (sqrt(pointNKNSquaredDistance[0]) >= 0.9*step)
            {
                break;
            }
            //std::cout << sqrt(pointNKNSquaredDistance[0]) <<std::endl;
            grid.x = cloud_projected->points[pointIdxNKNSearch[0]].x;
            grid.y = cloud_projected->points[pointIdxNKNSearch[0]].y;
            grid.z = cloud_projected->points[pointIdxNKNSearch[0]].z;
            point_index_2.push_back(pointIdxNKNSearch[0]);
            grid_point_3.push_back(grid);

            k++;
        }
        grid_point_3.clear();
        i++;
    } 
    grid_point.clear();
    std::cout<<"face point detected!"<<std::endl;
    //=================================================================
    //储存网格点索引
    //for (i=0;i<point_index.size();i++)
        //std::cout<< point_index[i] <<std::endl;
    //std::cout << point_index.size() <<std::endl;
    //红色网格点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_meshed (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_meshed->points.resize(point_index.size());
    cloud_meshed->width = point_index.size();
    cloud_meshed->height = 1;
    for (size_t i=0;i<point_index.size();i++)
    {
        cloud_meshed->points[i] = cloud_recon->points[point_index[i]];
    }
    pcl::io::savePCDFile("cloud_meshed.pcd",*cloud_meshed);

    //绿色面点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_meshed_2 (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_meshed_2->points.resize(point_index_2.size());
    cloud_meshed_2->width = point_index_2.size();
    cloud_meshed_2->height = 1;
    for (size_t i=0;i<point_index_2.size();i++)
    {
        cloud_meshed_2->points[i] = cloud_recon->points[point_index_2[i]];
    }
    std::cout<<"grid point detected!"<<std::endl;
    std::cout<<point_index.size()<<std::endl;
    //pcl::io::savePCDFile("mesh.pcd",*cloud_meshed);
    //===================================================================
    //去除point_index中的重复索引,改用哈希表更加快速
    std::vector<int> point_index_new;
    std::sort(point_index.begin(),point_index.end());
    point_index.erase(unique(point_index.begin(),point_index.end()),point_index.end());
    point_index_new = point_index;
    std::cout<< point_index_new.size() <<std::endl;

    std::vector<int> point_index_2_new;
    point_index_2_new.push_back(point_index_2[0]);
    std::sort(point_index_2.begin(),point_index_2.end());
    point_index_2.erase(unique(point_index_2.begin(),point_index_2.end()),point_index_2.end());
    point_index_2_new = point_index_2;
    
    std::cout<< "Repeat point removed!"<<std::endl;
    //===================================================================
    //手动取网格点
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
    kdtree2.setInputCloud(cloud_meshed);
    K=4;
    kdtree2.nearestKSearch(cloud_recon->points[point_index_2_new[999]],K,pointIdxNKNSearch,pointNKNSquaredDistance);
    
    //将该四角点放入一个点云中，后续拟合边界时使用
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_4corner (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_4corner->points.resize(4);
    cloud_4corner->width = 1;
    cloud_4corner->height = 4;
    cloud_4corner->points[0] = cloud_meshed->points[pointIdxNKNSearch[0]];
    cloud_4corner->points[1] = cloud_meshed->points[pointIdxNKNSearch[1]];
    cloud_4corner->points[2] = cloud_meshed->points[pointIdxNKNSearch[2]];
    cloud_4corner->points[3] = cloud_meshed->points[pointIdxNKNSearch[3]];

    //取该网格的三坐标范围
    float x_min=cloud_meshed->points[pointIdxNKNSearch[0]].x;
    float x_max=cloud_meshed->points[pointIdxNKNSearch[0]].x;
    float y_min=cloud_meshed->points[pointIdxNKNSearch[0]].y;
    float y_max=cloud_meshed->points[pointIdxNKNSearch[0]].y;
    float z_min=cloud_meshed->points[pointIdxNKNSearch[0]].z;
    float z_max=cloud_meshed->points[pointIdxNKNSearch[0]].z;
    for (size_t i=0;i<4;i++)
    {
        if (x_min > cloud_meshed->points[pointIdxNKNSearch[i]].x)
            x_min = cloud_meshed->points[pointIdxNKNSearch[i]].x;
        if (x_max < cloud_meshed->points[pointIdxNKNSearch[i]].x)
            x_max = cloud_meshed->points[pointIdxNKNSearch[i]].x;
        if (y_min > cloud_meshed->points[pointIdxNKNSearch[i]].y)
            y_min = cloud_meshed->points[pointIdxNKNSearch[i]].y;
        if (y_max < cloud_meshed->points[pointIdxNKNSearch[i]].y)
            y_max = cloud_meshed->points[pointIdxNKNSearch[i]].y;
        if (z_min > cloud_meshed->points[pointIdxNKNSearch[i]].z)
            z_min = cloud_meshed->points[pointIdxNKNSearch[i]].z;
        if (z_max < cloud_meshed->points[pointIdxNKNSearch[i]].z)
            z_max = cloud_meshed->points[pointIdxNKNSearch[i]].z;
    }
    std::cout<<"x : min  "<<x_min<<" max "<<x_max<<std::endl;
    std::cout<<"y : min  "<<y_min<<" max "<<y_max<<std::endl;
    std::cout<<"z : min  "<<z_min<<" max "<<z_max<<std::endl;

    pcl::io::savePCDFile("cloud_4corner.pcd",*cloud_4corner);
    //===================================================================
    //以该坐标范围为条件，建立条件滤波器，过滤出一个网格内的曲面片
    //定义字段条件，选出坐标范围内的点云
    //内存溢出
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>);
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("x",pcl::ComparisonOps::GT,x_min)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("x",pcl::ComparisonOps::LT,x_max)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("y",pcl::ComparisonOps::GT,y_min)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("y",pcl::ComparisonOps::LT,y_max)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("z",pcl::ComparisonOps::GT,z_min)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new 
        pcl::FieldComparison<pcl::PointXYZ>("z",pcl::ComparisonOps::LT,z_max)));
    //使用条件滤波
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud(cloud_recon);
    condrem.setKeepOrganized(false);//设置true则保持点云结构，其他点变为Nan，false则不保留
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_conditional(new pcl::PointCloud<pcl::PointXYZ>);
    //执行
    condrem.filter(*cloud_conditional);
    //对条件滤波后的面再进行一次细微分割,判断点是否在正方形内，使得点没有重叠部分
    for (int i=0;i<cloud_conditional->points.size();i++)
    {
        if (!isInterface(cloud_conditional->points[i],cloud_4corner))
            {
                //cloud_conditional->erase(index+i);
                cloud_conditional->points[i].x = NAN;
                cloud_conditional->points[i].y = NAN;
                cloud_conditional->points[i].z = NAN;
            }

    }
    //去除nan点
    vector<int> mapping;
    pcl::removeNaNFromPointCloud(*cloud_conditional,*cloud_conditional,mapping);
    //输出该点云的点数目
    std::cout << "after cutting size : "<< cloud_conditional->points.size()<<std::endl;
    pcl::io::savePCDFile("cloud_conditional.pcd",*cloud_conditional);
    //===================================================================
    //法线估计
    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne;
    ne.setInputCloud(cloud_conditional);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod (tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(10);
    ne.compute (*normals);
    std::cout<<"normal calculate finished! .."<< std::endl;
    //边界估计
    /* pcl::PointCloud<pcl::Boundary> boundary;
    pcl::BoundaryEstimation<pcl::PointXYZ,pcl::Normal,pcl::Boundary> est;
    est.setInputCloud(cloud_conditional);
    est.setInputNormals(normals);
    est.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
    est.setKSearch(80);
    est.compute(boundary);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (int i=0;i<cloud_conditional->points.size();i++)
    {
        if (boundary[i].boundary_point >0)
            cloud_boundary->push_back(cloud_conditional->points[i]);
    }
    std::cout<<"boundary calculate finished! .."<< std::endl;
    pcl::io::savePCDFile("boundary.pcd",*cloud_boundary); */
    //====================================================================
    //对点云进行nurbs建模
    pcl::on_nurbs::NurbsDataSurface data;
    PointCloud2Vector3d(cloud_conditional,data.interior);
    //PointCloud2Vector3d(cloud_boundary,data.boundary);//定义boundary
    unsigned order (3);//多项式阶数
    unsigned refinement (4);//细化迭代次数
    unsigned iterations (20);//迭代次数，完成细化后执行的迭代次数
    unsigned mesh_resolution (256);//每个参数化方向的顶点数，用于B样条曲面的三角剖分

    pcl::on_nurbs::FittingSurface::Parameter params;
    params.interior_smoothness = 0.01;//表面内部的平滑度(0.05)
    params.interior_weight = 1.0;//用于优化表面内部的权重
    params.boundary_smoothness = 0.1;//表面边界的平滑度
    params.boundary_weight = 1.0;//优化表面边界的权重

    std::cout<<" surface fitting ..."<<std::endl;
    ON_NurbsSurface nurbs = pcl::on_nurbs::FittingSurface::initNurbsPCABoundingBox (order,&data);
    pcl::on_nurbs::FittingSurface fit (&data,nurbs);
    // 可视化初始化。on_nurbs::Triangulation类允许NurbsSurface与PolygonmMesh互相转换，以可视化曲面
    pcl::PolygonMesh mesh;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::Vertices> mesh_vertices;
    std::string mesh_id = "mesh_nurbs";
    pcl::on_nurbs::Triangulation::convertSurface2PolygonMesh (fit.m_nurbs, mesh, mesh_resolution);
    viewer->addPolygonMesh (mesh, mesh_id); //添加曲面
    //表面细化
    Start=clock();
    for (size_t i = 0; i < refinement; i++)
    {
    fit.refine (0);
    fit.refine (1);// void refine：通过每两个节点之间插入中间节点修正曲线
    fit.assemble (params);// void assemble：根据约束条件，给定系统方程。对于大型点云来说，这一步是非常耗时的。
    fit.solve ();// void solve： 求解系统方程组
    pcl::on_nurbs::Triangulation::convertSurface2Vertices (fit.m_nurbs, mesh_cloud, mesh_vertices, mesh_resolution);
    viewer->updatePolygonMesh<pcl::PointXYZ> (mesh_cloud, mesh_vertices, mesh_id);
    viewer->spinOnce ();//调用一次交互器并更新一次屏幕。param中有time参数可以设置循环多久一次，单位为ms
    }
    // surface fitting with final refinement level 完成细化后执行迭代次数
    for (unsigned i = 0; i < iterations; i++)
    {
    fit.assemble (params);
    fit.solve ();
    pcl::on_nurbs::Triangulation::convertSurface2Vertices (fit.m_nurbs, mesh_cloud, mesh_vertices, mesh_resolution);
    viewer->updatePolygonMesh<pcl::PointXYZ> (mesh_cloud, mesh_vertices, mesh_id);//更新面网格数据
    viewer->spinOnce ();
    }
    End=clock();
    double endtime = (double)(End-Start)/CLOCKS_PER_SEC;
    std::cout<<"Total time: "<<endtime<<" s"<<std::endl;
    std::cout<<"fitting finished! ..."<<std::endl;
    std::cout<<"conditional cloud size: "<<cloud_conditional->points.size()<<std::endl;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_mesh_nurbs(new pcl::PointCloud<pcl::PointXYZ>);
    fromPCLPointCloud2(mesh.cloud,*cloud_mesh_nurbs);//将mesh变成点云
    //double err_nurbs = calculate_err_point(cloud_conditional,cloud_mesh_nurbs);//计算误差（点）
    //std::cout<< "nurbs err: "<<err_nurbs <<std::endl;
    pcl::io::savePCDFile("mesh_nurbs.pcd",*mesh_cloud);
    //==================================================================
    //用四边曲线对其进行裁剪
    //pcl::on_nurbs::NurbsDataSurface data_curve;
    PointCloud2Vector3d(cloud_4corner,data.interior);
    pcl::on_nurbs::FittingCurve2dAPDM::FitParameter curve_params;
    curve_params.addCPsAccuracy = 5e-2;//曲线的支持区域到最近数据点的距离必须低于该值，否则将插入控制点
    curve_params.addCPsIteration = 3;//内部迭代没有插入控制点
    curve_params.maxCPs = 4;//控制点的最大总数
    curve_params.accuracy = 1;//曲线的拟合精度
    curve_params.iterations = 10;//迭代次数

    curve_params.param.closest_point_resolution = 0;//每个支持区域内的控制点数量
    curve_params.param.closest_point_weight = 1.0;//曲线拟合到最近点的权值
    curve_params.param.closest_point_sigma2 = 0.1;//最近点的阈值
    curve_params.param.interior_sigma2 = 0.00001;//内点的阈值
    curve_params.param.smooth_concavity = 0;//导致曲线向内弯曲的值（0=不弯曲，<0向内弯曲，>0向外弯曲）
    curve_params.param.smoothness = 1.0;//平滑项的权值

    printf ("  curve fitting ...\n");
    pcl::on_nurbs::NurbsDataCurve2d curve_data;
    curve_data.interior = data.interior_param;
    curve_data.interior_weight_function.push_back (true);//为所有点启用了内部权重
    ON_NurbsCurve curve_nurbs = pcl::on_nurbs::FittingCurve2dAPDM::initNurbsCurve2D (order, curve_data.interior);

    // curve fitting 拟合曲线，曲线进行迭代拟合和细化
    pcl::on_nurbs::FittingCurve2dASDM curve_fit (&curve_data, curve_nurbs);
    // curve_fit.setQuiet (false); // enable/disable debug output
    curve_fit.fitting (curve_params);
    visualizeCurve (curve_fit.m_nurbs, fit.m_nurbs, viewer);

    //===================================================================
    //计算点到曲面的距离
    Eigen::Vector3d point_closest;
    Eigen::Vector2d hint(100,100);
    //int maxSteps=100;
    //double accuracy=1e-6;
    double error;
    Eigen::Vector3d tu;
    Eigen::Vector3d tv;
    Eigen::Vector2d point_p;
    pcl::PointCloud<PointXYZ>::Ptr xyz_cloud_filtered (new pcl::PointCloud<PointXYZ>);
    xyz_cloud_filtered->points.resize (cloud_conditional->points.size());
    xyz_cloud_filtered->header = cloud_conditional->header;
    xyz_cloud_filtered->width = cloud_conditional->width;
    xyz_cloud_filtered->height = cloud_conditional->height;

    double distance[cloud_conditional->points.size()];
    double temp_sum = 0;
    double temp_sum2 = 0;
    for (size_t point_i =0;point_i<cloud_conditional->points.size();++point_i)
    {
    Eigen::Vector3d point_pt(cloud_conditional->points[point_i].x,cloud_conditional->points[point_i].y,cloud_conditional->points[point_i].z);
    point_p = fit.inverseMapping (fit.m_nurbs,point_pt,hint,error,point_closest,tu,tv);
    xyz_cloud_filtered->points[point_i].x=point_closest(0);
    xyz_cloud_filtered->points[point_i].y=point_closest(1);
    xyz_cloud_filtered->points[point_i].z=point_closest(2);
    double x1=pow((cloud_conditional->points[point_i].x - xyz_cloud_filtered->points[point_i].x),2);
    double y1=pow((cloud_conditional->points[point_i].y - xyz_cloud_filtered->points[point_i].y),2);
    double z1=pow((cloud_conditional->points[point_i].z - xyz_cloud_filtered->points[point_i].z),2);
    distance[point_i]=sqrt(x1+y1+z1);
    //舍弃差别过大的点.可以用离群方差筛选来做过滤
    temp_sum += distance[point_i];
    temp_sum2 += distance[point_i]*distance[point_i]; 
    //cout<<"point "<< point_i<< " distance: "<<distance[point_i]<<endl;
    }
    pcl::io::savePCDFile("xyz_cloud_filtered.pcd",*xyz_cloud_filtered);

    double thre =1.0;//离群筛选阈值,减小会保留更少数。当该值为1.0时能够保留80%以上数据点
    double mean = temp_sum/cloud_conditional->points.size();
    double vari = temp_sum2/cloud_conditional->points.size() - mean*mean;//方差
    double temp_dis = 0;
    int count = 0;
    for (size_t i=0;i<cloud_conditional->points.size();i++)
    {
        if (distance[i] > mean-sqrt(vari)*thre && distance[i] < mean+sqrt(vari)*thre)
        {
            //cout<<distance[i]<<endl;
            count++;
            temp_dis +=distance[i];
        }
    }
    cout <<"average distance : "<<temp_dis/count<<endl;
    cout <<"count : "<<count <<endl; 

    //===================================================================
    //保存为3dm文件
    if (fit.m_nurbs.IsValid())
    {
        ONX_Model model;
        ONX_Model_Object& surf = model.m_object_table.AppendNew();
        surf.m_object = new ON_NurbsSurface(fit.m_nurbs);
        surf.m_bDeleteObject = true;
        surf.m_attributes.m_layer_index = 1;
        surf.m_attributes.m_name = "suface";

        model.Write("grid.3dm");
        std::cout<<" model saved : grid.3dm" <<std::endl;
    }
    //===================================================================
    //点云可视化
    //viewer->addPointCloud(cloud_recon,"data");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud_meshed,255,0,0);
    std::string cloudName="grid";
    viewer->addPointCloud(cloud_meshed,red,cloudName);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,cloudName);
    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud_meshed,0,255,0);
    std::string cloudName2="a grid";
    viewer->addPointCloud(cloud_meshed_2,green,cloudName2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,cloudName2);

    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloud_meshed,0,0,255);
    //std::string cloudName3="b grid";
    //viewer->addPointCloud(cloud_grid,blue,cloudName3);
    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,10,cloudName3); 

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloud_conditional,0,0,255);
    std::string cloudName3="b grid";
    //viewer->addPointCloud(xyz_cloud_filtered,blue,cloudName3);//蓝点为对应投影点
    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,cloudName3); 
    
    viewer->addPointCloud(cloud_conditional,blue,"data");//红点为原数据点
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"data"); 

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    } 
    //viewer->spin();
    return 0;

}

void
PointCloud2Vector3d (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::on_nurbs::vector_vec3d &data)//将点云数据压缩入一个3D向量的容器
{
  for (unsigned i = 0; i < cloud->size (); i++)
  {
    pcl::PointXYZ &p = cloud->at (i);
    if (!std::isnan (p.x) && !std::isnan (p.y) && !std::isnan (p.z))
      data.push_back (Eigen::Vector3d (p.x, p.y, p.z));
  }
}

void
visualizeCurve (ON_NurbsCurve &curve, ON_NurbsSurface &surface, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr curve_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::on_nurbs::Triangulation::convertCurve2PointCloud (curve, surface, curve_cloud, 4);
  for (std::size_t i = 0; i < curve_cloud->size () - 1; i++)
  {
    pcl::PointXYZRGB &p1 = curve_cloud->at (i);
    pcl::PointXYZRGB &p2 = curve_cloud->at (i + 1);
    std::ostringstream os;
    os << "line" << i;
    viewer->removeShape (os.str ());
    viewer->addLine<pcl::PointXYZRGB> (p1, p2, 1.0, 0.0, 0.0, os.str ());
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr curve_cps (new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int i = 0; i < curve.CVCount (); i++)
  {
    ON_3dPoint p1;
    curve.GetCV (i, p1);

    double pnt[3];
    surface.Evaluate (p1.x, p1.y, 0, 3, pnt);
    pcl::PointXYZRGB p2;
    p2.x = float (pnt[0]);
    p2.y = float (pnt[1]);
    p2.z = float (pnt[2]);

    p2.r = 255;
    p2.g = 0;
    p2.b = 0;

    curve_cps->push_back (p2);
  }
  viewer->removePointCloud ("cloud_cps");
  viewer->addPointCloud (curve_cps, "cloud_cps");
}

