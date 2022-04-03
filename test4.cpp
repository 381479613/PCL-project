#include <iostream>
#include <algorithm>
#include <time.h>
#include <eigen3/Eigen/Eigen>

#include <pcl/filters/conditional_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/on_nurbs/triangulation.h>
#include <pcl/surface/on_nurbs/fitting_surface_tdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/gp3.h>//贪婪三角化类
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <boost/thread/thread.hpp>
//#include <pcl/registration/icp.h>//ICP配准类相关头文件
#include "trimmed_icp_new.h"//tricp配准头文件
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/surface/poisson.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_search.h>

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_conditional (new pcl::PointCloud<pcl::PointXYZ>);
	  if (pcl::io::loadPCDFile("cloud_conditional.pcd", *cloud_conditional)<0)
    {
        PCL_ERROR("\\origin点云文件不存在!\\\n");
        system("pause");
        return -1;
    }
    std::cout<<"->加载点的个数:"<<cloud_conditional->points.size()<<std::endl;
    
    //为了获得某一条直线上的最邻近点
    //首先采用PCA确定直线方程，确定起始点以及步长，采用
    //int pcl::octree::OctreePointCloudSearch< PointT, LeafContainerT, BranchContainerT >::getIntersectedVoxelIndices  ( 
    //Eigen::Vector3f  origin,  
    //Eigen::Vector3f  direction,  
    //std::vector< int > &  k_indices,  
    //int  max_voxel_count = 0  
    //) 
    //确定起点与方向，返回八叉树体素与射线相交的体素的点的索引，k_indices
    //误差在于，小面片未必是一个平面片，需要去掉曲率部分
    //采用分割面片时计算出的特征向量作为主方向向量
    Eigen::Vector3f main_dir(-0.266987, -0.963503,-0.0195062);
    Eigen::Vector3f dice_dir(-0.963683,  0.267047,-0.00049591);
    Eigen::Vector3f origin(-5746.025391, -20728.953125, -292.169128);
    Eigen::Vector3f direction = dice_dir;
    std::vector<int> k_indices;
    //创建实例化八叉树对象
/*     int max_voxel_count=100;
    float resolution = 50.0f;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud_conditional);
    octree.addPointsFromInputCloud();
    octree.getIntersectedVoxelIndices (origin, direction, k_indices,  max_voxel_count);
    std::cout << k_indices.size()<< std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr line (new pcl::PointCloud<pcl::PointXYZ>);
    line->points.resize(k_indices.size());
    for (int i=0;i<k_indices.size();i++)
    {
        line->points[i]=cloud_conditional->points[k_indices[i]];
    } */
    //采用直线取最近邻半径点的算法
    float pt_squaredis=0.0f;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_conditional);
    int K = 1;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    pcl::PointXYZ point;
    point.x = origin.x();
    point.y = origin.y();
    point.z = origin.z();
    int step = 10;
    while (pt_squaredis < 25)
    {
        point.x -= step*direction.x();
        point.y -= step*direction.y();
        point.z -= step*direction.z();

        kdtree.nearestKSearch(point,K,pointIdxNKNSearch,pointNKNSquaredDistance);
        k_indices.push_back(pointIdxNKNSearch[0]);
        point = cloud_conditional->points[pointIdxNKNSearch[0]];
        pt_squaredis = pointNKNSquaredDistance[0];
        std::cout<<"squaredis: "<< pointNKNSquaredDistance[0]<<std::endl;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr line (new pcl::PointCloud<pcl::PointXYZ>);
    line->points.resize(k_indices.size());
    line->width = k_indices.size();
    line->height = 1;
    for (int i=0;i<k_indices.size();i++)
    {
        line->points[i]=cloud_conditional->points[k_indices[i]];
    }
    pcl::io::savePCDFile("line.pcd",*line);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("viewer"));
    //viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud_conditional,255,0,0);
    std::string cloudName="cloud_conditional";
    viewer->addPointCloud(cloud_conditional,red,cloudName);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,cloudName);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(line,0,0,255);
    std::string cloudName2="cloud_mesh";
    viewer->addPointCloud(line,blue,cloudName2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,cloudName2);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    } 

    return 0;
}
