// OpenCV can be used to read images.
#include <opencv2/opencv.hpp>



#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <climits>
#include <cfloat>
#include <vector>
#include <valarray>
#include <dirent.h>
#include <errno.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>
#include <memory>
#include <string>
//#include <tuple>
#include <iomanip>
//#include <vld.h>






// The VLFeat header files need to be declared external.
extern "C" {
    #include "vl/generic.h"
    #include "vl/slic.h"
    #include "vl/quickshift.h"
    #include "vl/random.h"
    #include "vl/mathop.h"
    #include "vl/stringop.h"
    #include "vl/host.h"
    #include "vl/imopv.h"
    #include "vl/pgm.h"
    #include "vl/rodrigues.h"
    #include "vl/sift.h"
    #include "vl/fisher.h"
    #include "vl/gmm.h"
    #include "vl/svm.h"
    #include "vl/kmeans.h"
}

using namespace std;

vector<vector<vl_sift_pix> > sift_descr(vl_sift_pix* image, int largeur_image, int hauteur_image, int nb_octaves, int nb_niveaux,
	int octave_init, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region,
	double taille_fenetre )
{

		//Declaration des differents parametres
		double angles[4];
		vector<vector<vl_sift_pix> > ensemble_descripteur;
		vector<vl_sift_pix> temp;
		vl_sift_pix *descripteur = (float*)calloc(128,sizeof(float));
		int nb_orientations;
		int nb_pointscle;
		VlSiftKeypoint const **pointscle = (const VlSiftKeypoint**)calloc(nb_octaves,sizeof(VlSiftKeypoint*));

		//Creation de l objet filtre SIFT
		VlSiftFilt *filtresift = vl_sift_new(largeur_image, hauteur_image, nb_octaves, nb_niveaux, octave_init);

		//On affectation a l objet filtre SIFT les parametres lies a la description d un point cle 
		vl_sift_set_magnif(filtresift, taille_region);
		vl_sift_set_window_size(filtresift, taille_fenetre);

		//On affectation a l objet filtre SIFT les parametres lies a la validation d un point cle 
		vl_sift_set_edge_thresh(filtresift,seuil_courbure_contours);
		vl_sift_set_peak_thresh(filtresift,seuil_extrema_locaux);

		//Pour chaque octave de l esapce d echelles :
		for(int k = 0 ; k < nb_octaves ; k++)
		{

			//On determine les fonctions DOG
			//Appel d une fonction particuliere pour la premiere octave
			if(k == 0)
			{

				vl_sift_process_first_octave(filtresift,image);

			}

			else
			{

				vl_sift_process_next_octave(filtresift);

			}

			//On calcul et on stock l ensemble des points cle pour l octave courante
			vl_sift_detect(filtresift);
			pointscle[k] =  vl_sift_get_keypoints(filtresift);

			//Obtention du nombre de point cle detecte pour l octave courante
			nb_pointscle = vl_sift_get_nkeypoints(filtresift);

			//Pour chaque point cle :
			for(int i = 0 ; i <nb_pointscle ; i++ )
			{

				//Obtention du nombre d orientation du point cle considere (minimum 1)
				nb_orientations = vl_sift_calc_keypoint_orientations(filtresift,angles,(pointscle[k]+i));

				//Pour chaque orientation :
				for(int j = 0; j<nb_orientations; j++)
				{

					//On calcul le descripteur associe
					vl_sift_calc_keypoint_descriptor(filtresift, descripteur, (pointscle[k]+i), angles[j]);

					temp.clear();

					//On l'ajoute a la variable que l on va retourner
					for(int l = 0; l < 128 ; l++)
					{

						temp.push_back(descripteur[l]);

					}

					ensemble_descripteur.push_back(temp);

				}
			}
		}

		//Suppression de l objet filtresift
		vl_sift_delete(filtresift);

		free(pointscle);
		free(descripteur);

		return ensemble_descripteur;

}



int main() {
    // Read the Lenna image. The matrix 'mat' will have 3 8 bit channels
    // corresponding to BGR color space.
    cv::Mat mat = cv::imread("Lenna.png", CV_LOAD_IMAGE_COLOR);
    
    // Convert image to one-dimensional array.
    float* image = new float[mat.rows*mat.cols*mat.channels()];
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            // Assuming three channels ...
            image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
            image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
            image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
        }
    }

    // The algorithm will store the final segmentation in a one-dimensional array.
    vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
    vl_size height = mat.rows;
    vl_size width = mat.cols;
    vl_size channels = mat.channels();
            
    // The region size defines the number of superpixels obtained.
    // Regularization describes a trade-off between the color term and the
    // spatial term.
    vl_size region = 30;        
    float regularization = 1000.;
    vl_size minRegion = 10;
          
    vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);
            
    // Convert segmentation.
    int** labels = new int*[mat.rows];
    for (int i = 0; i < mat.rows; ++i) {
        labels[i] = new int[mat.cols];
                
        for (int j = 0; j < mat.cols; ++j) {
            labels[i][j] = (int) segmentation[j + mat.cols*i];
        }
    }
    
    int label = 0;
    int labelTop = -1;
    int labelBottom = -1;
    int labelLeft = -1;
    int labelRight = -1;
    
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            
            label = labels[i][j];
            
            labelTop = label;
            if (i > 0) {
                labelTop = labels[i - 1][j];
            }
            
            labelBottom = label;
            if (i < mat.rows - 1) {
                labelBottom = labels[i + 1][j];
            }
            
            labelLeft = label;
            if (j > 0) {
                labelLeft = labels[i][j - 1];
            }
            
            labelRight = label;
            if (j < mat.cols - 1) {
                labelRight = labels[i][j + 1];
            }
            
            if (label != labelTop || label != labelBottom || label!= labelLeft || label != labelRight) {
                mat.at<cv::Vec3b>(i, j)[0] = 0;
                mat.at<cv::Vec3b>(i, j)[1] = 0;
                mat.at<cv::Vec3b>(i, j)[2] = 255;
            }
        }
    }
    
    cv::imwrite("Lenna_contours.png", mat);
    
    return 0;
}
