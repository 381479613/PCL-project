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

double calculate_err(pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_origin,pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud_triangle)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(_cloud_triangle);
  double result=0;
  for (auto points: _cloud_origin->points)
  {
    int K=3;//寻找最近的三个点
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    kdtree.nearestKSearch(points,K,pointIdxNKNSearch,pointNKNSquaredDistance);
    Eigen::Vector3f v_1,v_2,v_3;
    v_1.x() = _cloud_triangle->points[pointIdxNKNSearch[0]].x;
    v_1.y() = _cloud_triangle->points[pointIdxNKNSearch[0]].y;
    v_1.z() = _cloud_triangle->points[pointIdxNKNSearch[0]].z;
    v_2.x() = _cloud_triangle->points[pointIdxNKNSearch[1]].x;
    v_2.y() = _cloud_triangle->points[pointIdxNKNSearch[1]].y;
    v_2.z() = _cloud_triangle->points[pointIdxNKNSearch[1]].z;
    v_3.x() = _cloud_triangle->points[pointIdxNKNSearch[2]].x;
    v_3.y() = _cloud_triangle->points[pointIdxNKNSearch[2]].y;
    v_3.z() = _cloud_triangle->points[pointIdxNKNSearch[2]].z;
    Eigen::Vector3f cross_res = (v_1-v_2).cross(v_1-v_3);
    double d = -((cross_res.cwiseProduct(v_1)).x()+(cross_res.cwiseProduct(v_1)).y()+(cross_res.cwiseProduct(v_1)).z());
    double t = abs(cross_res.x()*points.x+cross_res.y()*points.y+cross_res.z()*points.z)/sqrt((cross_res.x()*cross_res.x()+cross_res.y()*cross_res.y()+cross_res.z()*cross_res.z()));
    cout << "t :"<<t<<endl;
    Eigen::Vector3f proj_point;
    proj_point.x() = points.x-cross_res.x()*t;
    proj_point.y() = points.y-cross_res.y()*t;
    proj_point.z() = points.z-cross_res.z()*t;//计算点在投影面上的坐标
    Eigen::Vector3f vec_tmp1 = (v_2-v_1).cross(proj_point-v_1);
    Eigen::Vector3f vec_tmp2 = (v_3-v_2).cross(proj_point-v_2);
    Eigen::Vector3f vec_tmp3 = (v_1-v_3).cross(proj_point-v_3);
    if (vec_tmp1.dot(vec_tmp2)>0 && vec_tmp1.dot(vec_tmp3)>0 && vec_tmp2.dot(vec_tmp3)>0)
      result += t;
    else if (vec_tmp1.dot(vec_tmp2)<0 && vec_tmp1.dot(vec_tmp3)<0 && vec_tmp2.dot(vec_tmp3)<0)
      result += t;
    else 
      {
        Eigen::Vector3f this_points;
        this_points.x()=points.x;
        this_points.y()=points.y;
        this_points.z()=points.z;
        /* Eigen::Vector3f v1tov2 = v_2-v_1;
        Eigen::Vector3f v2tov3 = v_3-v_2;
        Eigen::Vector3f v3tov1 = v_1-v_3;
        double dis_1 = (v1tov2.cross(this_points-v_1)).norm()/v1tov2.norm();
        double dis_2 = (v2tov3.cross(this_points-v_2)).norm()/v2tov3.norm();
        double dis_3 = (v3tov1.cross(this_points-v_3)).norm()/v3tov1.norm(); */
        double dis_1 = (this_points-v_1).norm();
        double dis_2 = (this_points-v_2).norm();
        double dis_3 = (this_points-v_3).norm();
        result += std::min(dis_1,std::min(dis_2,dis_3));
      }
  }
  result = result / _cloud_origin->points.size();
  return result;
}

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
    
  /*   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_conditional_2 (new pcl::PointCloud<pcl::PointXYZ>);
	  if (pcl::io::loadPCDFile("cloud_conditional_2.pcd", *cloud_conditional_2)<0)
    {
        PCL_ERROR("\\origin点云文件不存在!\\\n");
        system("pause");
        return -1;
    }
    std::cout<<"->加载点的个数:"<<cloud_conditional_2->points.size()<<std::endl; */
    //===================================================================
    //法线估计
    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne;
    ne.setInputCloud(cloud_conditional);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(20);
    ne.compute (*normals);
    //将法线数据并入点云数据中
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud_conditional,*normals,*cloud_with_normals);
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);
    //poisson重建
    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setInputCloud(cloud_with_normals);
    poisson.setConfidence(false);
    poisson.setDegree(2);//阶数，越大越清晰
    poisson.setDepth(8);//树的最大深度
    poisson.setIsoDivide(8);//提取iso等值面的算法深度
    poisson.setOutputPolygons(false);//输出多边形网格
    poisson.setSamplesPerNode(3.0);//八叉树结点中的样本点最小数量。1-5 无噪声 15-20有噪声
    poisson.setScale(1.05);//重构的立方体直径和样本边界立方体直径的比率
    poisson.setSolverDivide(8);//求解线性方程组的Gauss-Seidel迭代方法深度
    pcl::PolygonMesh mesh;
    poisson.performReconstruction(mesh);
    pcl::io::savePLYFile("mesh.ply",mesh);
    pcl::io::saveVTKFile("mesh.vtk",mesh);
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("Viewer"));

    std::string mesh_id="poisson_1";
    //viewer->addPolygonMesh (mesh, mesh_id); //添加曲面

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_mesh(new pcl::PointCloud<pcl::PointXYZ>);
    fromPCLPointCloud2(mesh.cloud,*cloud_mesh);//将mesh变成点云
    std::cout<<"重建后点个数："<<cloud_mesh->points.size()<<std::endl;

    double err_poisson = calculate_err(cloud_conditional,cloud_mesh);//计算误差（三角形面）
    //double err_poisson = calculate_err_point(cloud_conditional,cloud_mesh);//计算误差（点）

    std::cout<< "poisson err: "<<err_poisson <<std::endl;

    /* viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud_conditional,255,0,0);
    std::string cloudName="cloud_conditional";
    viewer->addPointCloud(cloud_conditional,red,cloudName);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,cloudName);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloud_mesh,0,0,255);
    std::string cloudName2="cloud_mesh";
    viewer->addPointCloud(cloud_mesh,blue,cloudName2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1,cloudName2); */
    
    //贪婪三角化划分
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setSearchRadius (30);//三角形最大边长
    gp3.setMu (50);//指定相对最近邻点距离可以接受的最大距离
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates(); 
    std::string mesh_id_gp = "gp";
    viewer->addPolygonMesh (triangles, mesh_id_gp); //添加曲面

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_mesh_gp(new pcl::PointCloud<pcl::PointXYZ>);
    fromPCLPointCloud2(triangles.cloud,*cloud_mesh_gp);//将mesh变成点云
    std::cout<<"重建后点个数："<<cloud_mesh_gp->points.size()<<std::endl;
    
    double err_gp = calculate_err_point(cloud_conditional,cloud_mesh_gp);//计算误差（点）

    std::cout<< "gp err: "<<err_gp <<std::endl;

  
    //点云配准
    /* pcl::PointCloud<pcl::PointXYZ>::Ptr CadModel (new pcl::PointCloud<pcl::PointXYZ>);
	  if (pcl::io::loadPCDFile("CadModel.pcd", *CadModel)<0)
    {
        PCL_ERROR("\\Cad Modeln点云文件不存在!\\\n");
        system("pause");
        return -1;
    }

    pcl::recognition::TrimmedICP<pcl::PointXYZ,double> tricp;
    tricp.init(CadModel);
    Eigen::Matrix4d transformation_matrix=Eigen::Matrix4d::Identity();
    clock_t start =  clock();
    tricp.align(*cloud_conditional,(int)cloud_conditional->size(),transformation_matrix);
    clock_t end=clock();
    cout << "time:" << (double)(end - start) / (double)CLOCKS_PER_SEC << " s" << endl;
	  cout << "matrix:" << endl << transformation_matrix << endl << endl << endl;
	  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_end(new pcl::PointCloud<pcl::PointXYZ>);
  	pcl::transformPointCloud(*cloud_conditional, *cloud_end, transformation_matrix);

    viewer->addCoordinateSystem(1.0); */

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    } 

    return 0;
}
