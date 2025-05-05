#include "features_matcher.h"

#include <iostream>
#include <map>

FeatureMatcher::FeatureMatcher(cv::Mat intrinsics_matrix, cv::Mat dist_coeffs, double focal_scale)
{
  intrinsics_matrix_ = intrinsics_matrix.clone();
  dist_coeffs_ = dist_coeffs.clone();
  new_intrinsics_matrix_ = intrinsics_matrix.clone();
  new_intrinsics_matrix_.at<double>(0,0) *= focal_scale;
  new_intrinsics_matrix_.at<double>(1,1) *= focal_scale;
}

cv::Mat FeatureMatcher::readUndistortedImage(const std::string& filename )
{
  cv::Mat img = cv::imread(filename), und_img, dbg_img;
  cv::undistort	(	img, und_img, intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_ );

  return und_img;
}

void FeatureMatcher::extractFeatures()
{
  features_.resize(images_names_.size());
  descriptors_.resize(images_names_.size());
  feats_colors_.resize(images_names_.size());

  //Object for SIFT detector
  cv::Ptr<cv::SIFT> sift_detector = cv::SIFT::create();

  for( int i = 0; i < images_names_.size(); i++  )
  {
    std::cout<<"Computing descriptors for image "<<i<<std::endl;
    cv::Mat img = readUndistortedImage(images_names_[i]);

    //////////////////////////// Code to be completed (1/7) /////////////////////////////////
    // Extract salient points + descriptors from i-th image, and store them into
    // features_[i] and descriptors_[i] vector, respectively
    // Extract also the color (i.e., the cv::Vec3b information) of each feature, and store
    // it into feats_colors_[i] vector
    /////////////////////////////////////////////////////////////////////////////////////////

     //Color vector
     std::vector<cv::Vec3b> tmp_colors;
     //Keypoint vector
     std::vector<cv::KeyPoint> tmp_keypoints;
     // Descriptor vector
     cv::Mat tmp_descriptor;
     sift_detector->detectAndCompute(img, cv::noArray(), tmp_keypoints,tmp_descriptor);
    
    features_[i] = tmp_keypoints;
    descriptors_[i] = tmp_descriptor;

    for (const auto& kp : tmp_keypoints)
    {
      int x = cvRound(kp.pt.x);
      int y = cvRound(kp.pt.y);

      //Verifica che il punto sia dentro l'immagine
      if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
      {
        tmp_colors.push_back(img.at<cv::Vec3b>(y, x));
      }
      else
      {
        tmp_colors.push_back(cv::Vec3b(0, 0, 0));
      }
    }
    feats_colors_[i] = tmp_colors;
    
    /////////////////////////////////////////////////////////////////////////////////////////
  }
}

void FeatureMatcher::exhaustiveMatching()
{
  std::vector<cv::DMatch> matches, inlier_matches;

  
  for( int i = 0; i < images_names_.size() - 1; i++ )
  {
    for( int j = i + 1; j < images_names_.size(); j++ )
    {
      std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;

      //////////////////////////// Code to be completed (2/7) /////////////////////////////////
      // Match descriptors between image i and image j, and perform geometric validation,
      // possibly discarding the outliers (remember that features have been extracted
      // from undistorted images that now has new_intrinsics_matrix_ as K matrix and
      // no distortions)
      // As geometric models, use both the Essential matrix and the Homograph matrix,
      // both by setting new_intrinsics_matrix_ as K matrix.
      // As threshold in the functions to estimate both models, you may use 1.0 or similar.
      // Store inlier matches into the inlier_matches vector
      // Do not set matches between two images if the amount of inliers matches
      // (i.e., geomatrically verified matches) is small (say <= 5 matches)
      // In case of success, set the matches with the function:
      // setMatches( i, j, inlier_matches);
      /////////////////////////////////////////////////////////////////////////////////////////
      
      // Create the matcher
      //cv::BFMatcher matcher(cv::NORM_L2, false);
      cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

      //L2 Normalization
      cv::Mat desc_norm1, desc_norm2;
      cv::normalize(descriptors_[i], desc_norm1, 1.0, 0.0, cv::NORM_L2);
      cv::normalize(descriptors_[j], desc_norm2, 1.0, 0.0, cv::NORM_L2);


      //Match descriptors
      matcher->match(desc_norm1, desc_norm2, matches);

      //Filter matches using the Lowe's ratio test
      double ratio_thresh = 0.7;
      //Lowe's ratio test con knnMatch
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(desc_norm1, desc_norm2, knn_matches, 2);

      matches.clear();
      for (size_t k = 0; k < knn_matches.size(); k++)
      {
        //Avoid illegal access to the memory
        if (knn_matches[k].size() == 2)
        {
          cv::DMatch& knn1 = knn_matches[k][0];
          cv::DMatch& knn2 = knn_matches[k][1];
          if (knn1.distance < ratio_thresh * knn2.distance)
          {
            matches.push_back(knn1);
          }
        }
      }

      // Convert keypoints to cv::Point2f
      std::vector<cv::Point2f> points1, points2;
      for (size_t k = 0; k < matches.size(); k++)
      {
        points1.push_back(features_[i][matches[k].queryIdx].pt);
        points2.push_back(features_[j][matches[k].trainIdx].pt);
      }

      // Check if there are enough matches
      if (points1.size() < 5 || points2.size() < 5)
      {
        std::cout << "Not enough matches found between image " << i << " and image " << j << std::endl;
        continue;
      }

      // Recover the pose from the Essential matrix
      cv::Mat R, t, E;

      //Essential matrix
      cv::Mat ess_matrix = cv::findEssentialMat(points1, points2, new_intrinsics_matrix_, cv::RANSAC, 0.999, 1.0, E);

      if (ess_matrix.empty()) {
        std::cout << "Essential matrix estimation failed between image " << i << " and " << j << std::endl;
        return;
      }else{
        std::cout << "Essential matrix estimation succeeded between image " << i << " and " << j << std::endl;
      }
    
      cv::recoverPose(ess_matrix, points1, points2,new_intrinsics_matrix_, R, t, E);

      // Estimate the Homography matrix
      cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 1.0);

      // Check if the Homography matrix is valid
      if (H.empty())
      {
        std::cout << "Homography estimation failed between image " << i << " and image " << j << std::endl;
        continue;
      }

      // Filter matches using the Homography matrix
      for (size_t k = 0; k < matches.size(); k++)
      {
        if (E.at<uchar>(k))
        {
          inlier_matches.push_back(matches[k]);
        }
      }
      // Check if there are enough inlier matches
      if (inlier_matches.size() < 5)
      {
        std::cout << "Not enough inlier matches found between image " << i << " and image " << j << std::endl;
        continue;
      }
      // Set the matches
      setMatches(i, j, inlier_matches);
      std::cout << "Matches set between image " << i << " and image " << j << std::endl;
      /////////////////////////////////////////////////////////////////////////////////////////

    }
  }
}

void FeatureMatcher::writeToFile ( const std::string& filename, bool normalize_points ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

  double *tmp_observations;
  cv::Mat dst_pts;
  if(normalize_points)
  {
    cv::Mat src_obs( num_observations_,1, cv::traits::Type<cv::Vec2d>::value,
                     const_cast<double *>(observations_.data()));
    cv::undistortPoints(src_obs, dst_pts, new_intrinsics_matrix_, cv::Mat());
    tmp_observations = reinterpret_cast<double *>(dst_pts.data);
  }
  else
  {
    tmp_observations = const_cast<double *>(observations_.data());
  }

  for (int i = 0; i < num_observations_; ++i)
  {
    fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", tmp_observations[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if( colors_.size() == 3*num_points_ )
  {
    for (int i = 0; i < num_points_; ++i)
      fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
  }

  fclose(fptr);
}

void FeatureMatcher::testMatches( double scale )
{
  // For each pose, prepare a map that reports the pairs [point index, observation index]
  std::vector< std::map<int,int> > cam_observation( num_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation[i_cam][i_pt] = i_obs;
  }

  for( int r = 0; r < num_poses_; r++ )
  {
    for (int c = r + 1; c < num_poses_; c++)
    {
      int num_mathces = 0;
      std::vector<cv::DMatch> matches;
      std::vector<cv::KeyPoint> features0, features1;
      for (auto const &co_iter: cam_observation[r])
      {
        if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
        {
          features0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1], 0.0);
          features1.emplace_back(observations_[2*cam_observation[c][co_iter.first]],observations_[2*cam_observation[c][co_iter.first] + 1], 0.0);
          matches.emplace_back(num_mathces,num_mathces, 0);
          num_mathces++;
        }
      }
      cv::Mat img0 = readUndistortedImage(images_names_[r]),
          img1 = readUndistortedImage(images_names_[c]),
          dbg_img;

      cv::drawMatches(img0, features0, img1, features1, matches, dbg_img);
      cv::resize(dbg_img, dbg_img, cv::Size(), scale, scale);
      cv::imshow("", dbg_img);
      if (cv::waitKey() == 27)
        return;
    }
  }
}

void FeatureMatcher::setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches )
{

  const auto &features0 = features_[pos0_id];
  const auto &features1 = features_[pos1_id];

  auto pos_iter0 = pose_id_map_.find(pos0_id),
      pos_iter1 = pose_id_map_.find(pos1_id);

  // Already included position?
  if( pos_iter0 == pose_id_map_.end() )
  {
    pose_id_map_[pos0_id] = num_poses_;
    pos0_id = num_poses_++;
  }
  else
    pos0_id = pose_id_map_[pos0_id];

  // Already included position?
  if( pos_iter1 == pose_id_map_.end() )
  {
    pose_id_map_[pos1_id] = num_poses_;
    pos1_id = num_poses_++;
  }
  else
    pos1_id = pose_id_map_[pos1_id];

  for( auto &match:matches)
  {

    // Already included observations?
    uint64_t obs_id0 = poseFeatPairID(pos0_id, match.queryIdx ),
        obs_id1 = poseFeatPairID(pos1_id, match.trainIdx );
    auto pt_iter0 = point_id_map_.find(obs_id0),
        pt_iter1 = point_id_map_.find(obs_id1);
    // New point
    if( pt_iter0 == point_id_map_.end() && pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = num_points_++;
      point_id_map_[obs_id0] = point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);

      // Average color between two corresponding features (suboptimal since we shouls also consider
      // the other observations of the same point in the other images)
      cv::Vec3f color = (cv::Vec3f(feats_colors_[pos0_id][match.queryIdx]) +
                        cv::Vec3f(feats_colors_[pos1_id][match.trainIdx]))/2;

      colors_.push_back(cvRound(color[2]));
      colors_.push_back(cvRound(color[1]));
      colors_.push_back(cvRound(color[0]));

      num_observations_++;
      num_observations_++;
    }
      // New observation
    else if( pt_iter0 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id1];
      point_id_map_[obs_id0] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      num_observations_++;
    }
    else if( pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id0];
      point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);
      num_observations_++;
    }
//    else if( pt_iter0->second != pt_iter1->second )
//    {
//      std::cerr<<"Shared observations does not share 3D point!"<<std::endl;
//    }
  }
}
void FeatureMatcher::reset()
{
  point_index_.clear();
  pose_index_.clear();
  observations_.clear();
  colors_.clear();

  num_poses_ = num_points_ = num_observations_ = 0;
}
