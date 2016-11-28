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

#include "generic.h"
#include "quickshift.h"
#include "random.h"
#include "mathop.h"
#include "stringop.h"
#include "host.h"
#include "imopv.h"
#include "pgm.h"
#include "rodrigues.h"
#include "sift.h"
#include "fisher.h"
#include "gmm.h"
#include "svm.h"
#include "kmeans.h"

using namespace cv;
using namespace std;

#define RAS 0x0000 //rien a signaler
#define err1 0x0002 // la fonction ensemble_fichier n a pas trouve le repertoire en entree
#define err2 0x0003 // la fonction nombre_fichier n a pas trouve le repertoire en entree
#define err3 0x0004 // la fonction lecture_fichier n a pas trouve le repertoire en entree
#define err4 0x0005 // la fonction lecture_fichier_vect n a pas trouve le repertoire en entree
#define err5 0x0006 // la fonction lecture_fichier_bin n a pas trouve le repertoire en entree
#define err6 0x0007 // la fonction lecture_fichier_bin2D n a pas trouve le repertoire en entree
#define err7 0x0008 // la fonction ecriture_fichier_bin2D a un vecteur de taille nulle en entree

//Pas vraiment sur de pourquoi j ai du declarer cette fonction qui permet de convertir un entier en string, un truc a voir avec C++11
std::string to_string(int i)
{
	std::stringstream ss;
	ss << i;
	return ss.str();
}

int maxi(int a, int b)
{
	if(a<b)
	{

		return b;

	}

	else
	{

		return a;

	}
}


//Voir http://www.vlfeat.org/api/sift.html pour mieux comprendre à quoi correspondent les parametres des fonction de la bibliotheque vl-feat
//Je ne rentrerai dans le detail lors de mes commentaires uniquement lorsque je traiterai de methodes utilisant des fonction de la bibliotheque
//vl-feat

//Retourne un descripteur SIFT de l image en entree : vl_sift_pix est un float et chaque vector<vl_sift_pix> correspond au descripteur d un point
//cle de notre image. Le vecteur renvoye est l ensemble des descripteurs des points cle.

//nb_octaves correspond au nombre d octaves dans l espace d echelle : un nombre eleve d octaves entrainera un plus grand nombre de points cle par
//image sauf si les images ont des dimensions faibles (<50 pixels en hauteur et en largeur), dans ce cas, au dela de 4 octaves, aucun nouveau point
//cle ne sera detecte

//nb_niveaux correspond au nombre de niveaux entre chaques octave dans l espace d echelles un nombre eleve de niveaux entre les octaves entrainera
//plus de points cle detectes

//octave_init permet de determiner le "flou" initial de l image. A fixer a 0 si l'on souhaite commencer l esapce d echelle avec l image originale,
//au dessus de 0 si on souhaite commencer l esapce d echelle avec une image filtree a l aide d un noyau gaussien de variance egale a octave_init
//et egal a -1 en cas de "petite image" pour commencer l espace dechelle avec une image surechantilonnee par un facteur 2 (en general on ne descend
//pas en dessous de la valeur -1)

//seuil_courbure_contours correspond a la valeur de la courbure de la fonction associes a l image a partir de laquelle on rejette un point cle
//a diminuer si l on souhaite diminuer le nombre de point cle

//seuil_extrema_locaux correspond a la valeur de la fonction DOG associes a l image a partir de laquelle on conserve un point cle
//a augmenter si l on souhaite diminuer le nombre de point cle

//taille_region correspond a la taille de la region permettant de decrire le voisinnage d un point cle

//taille_fenetre correspond a la variance du noyau de convolution gaussien utilise sur la region autour d un point cle. Une valeur faible donnera
//plus d importance aux pixels situes pres du centre

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



//Convertit l image en entree en vecteur de float

vector<float> transf_vect(Mat *im)
{

    vector<float> temp;

    for(int i = 0 ; i < im->rows; i++)
	{

        for(int j = 0 ; j < im->cols; j++)
		{

            temp.push_back( im->at<float>(i,j) );

        }

    }

    return temp;

}


//Convertit l image en entree en pointeur sur float

float* transf_point(Mat *im)
{

    float *temp = (float*)calloc(im->rows*im->cols,sizeof(float));
    int k = 0;

    for(int i = 0 ; i < im->rows; i++)
	{

        for(int j = 0 ; j < im->cols; j++)
		{

            temp[k] = im->at<float>(i,j);
            k++;

        }

    }

    return temp;

}


//Convertit le vecteur  en entree en pointeur

template<typename T>
T* vect_en_point(vector<T>  *descr)
{

	T *temp = (T*)calloc(descr->size(),sizeof(T));
	int k = 0;

	for(unsigned int i = 0 ; i < descr->size(); i++)
	{

		temp[k] = descr->at(i);
		k++;

	}

	return temp;

}





//Convertit le vecteur 2d en entree en pointeur

template<typename T>
T* vect2_en_point(vector<vector<T> > *descr)
{

	T *temp = (T*)calloc(descr->size()*descr->at(0).size(),sizeof(T));
	int k = 0;

	for(unsigned int i = 0 ; i < descr->size(); i++)
	{

		for(unsigned int j = 0 ; j < descr->at(i).size(); j++)
		{

			temp[k] = descr->at(i)[j];
			k++;

		}

	}

	return temp;

}



//Transforme le pointeur en entree en vecteur : ne fonctionne pas si on ne dispose pas du nombre d elements

template<typename T>
vector<T> point_en_vect(T *point, int taille)
{

	vector<T> res;

	for(int i = 0 ; i < taille ; i++)
	{

		res.push_back(point[i]);

	}

	return res;

}

//Retourne un objet im correspondant a l objet original ayant subit une rotation (dans le sens trigo) d angle (en degree) en entree

Mat rotation(Mat *im, double angle)
{

	Mat res;
	Point2d pt(im->cols/2., im->rows/2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(*im, res, r, Size(im->cols, im->rows));
	return res;

}


//retourne un vecteur de string contenant l'ensemble des chemins et noms des fichiers contenus dans le repertoire localise dans la variable chemin
//Ne fonctionne que s il n y a pas de repertoire dans le repertoire en entree

vector<string> ensemble_fichier(const char* chemin)
{

	DIR* rep = NULL;
	struct dirent* fichierLu = NULL;
	string chemin_rep(chemin);
	string temp;
	vector<string> ensemble_nom_images;

	rep = opendir(chemin);

	if(rep == NULL)
	{

		throw int(err1);

	}

	fichierLu = readdir(rep);
	fichierLu = readdir(rep);

	while ((fichierLu = readdir(rep)) != NULL)
	{

		temp = chemin;
		temp += "\\";
		temp += fichierLu->d_name;
		ensemble_nom_images.push_back(temp);

	}

	if(closedir(rep) == -1)
	{

		exit(-1);

	}

	free(fichierLu);

	return ensemble_nom_images;
}



//retourne le nombre de fichiers contenus dans le dossier localise dans la variable chemin

int nombre_fichier(const char* chemin)
{

	int compteur = 0;
	DIR* rep = NULL;
	struct dirent* fichierLu = NULL;

	rep = opendir(chemin);

	if(rep == NULL)
	{

		throw int(err2);
	}

	fichierLu = readdir(rep);
	fichierLu = readdir(rep);

	while ((fichierLu = readdir(rep)) != NULL)
	{

		compteur++;
	}

	if(closedir(rep) == -1)
	{

		exit(-1);

	}

	free(fichierLu);

	return compteur;
}



void ecriture_fichier_bin_int(vector<int> *var, string nom_fichier)
{

	FILE*f=fopen(nom_fichier.c_str(),"wb");

	for(unsigned int i = 0 ; i < var->size(); i++)
	{

		fwrite(&(var->at(i)),sizeof(var->at(i)),1,f);

	}

	fclose(f);

}

vector<int> lecture_fichier_bin_int(string nom_fichier)
{

	vector<int> data;

	int temp;

	size_t readElts;

	FILE* f = fopen(nom_fichier.c_str(),"rb");

	if(f == NULL)
	{

		throw int(err5);

	}

	do
	{

		readElts = fread(&temp,sizeof(temp),1,f);

		if(readElts)
		{

			data.push_back(temp);

		}

	}

	while(readElts);

	fclose(f);

	return data;

}


void ecriture_fichier_bin_double(vector<double> *var, string nom_fichier)
{


	FILE*f=fopen(nom_fichier.c_str(),"wb");

	for(unsigned int i = 0 ; i < var->size(); i++)
	{

		fwrite(&(var->at(i) ),sizeof(var->at(i)),1,f);

	}

	fclose(f);

}


vector<double> lecture_fichier_bin_double(string nom_fichier)
{

	vector<double> data;

	double temp;

	size_t readElts;

	FILE* f = fopen(nom_fichier.c_str(),"rb");

	if(f == NULL)
	{

		throw int(err5);

	}

	do
	{

		readElts = fread(&temp,sizeof(temp),1,f);

		if(readElts)
		{

			data.push_back(temp);

		}

	}

	while(readElts);

	fclose(f);

	return data;

}


void ecriture_fichier_bin2D(vector<vector<float>> *var, string nom_fichier)
{

	FILE*f=fopen(nom_fichier.c_str(),"wb");

	unsigned int nb_vect = var->size();

	if(nb_vect == 0)
	{
	
		throw int(err7);
	
	}

	unsigned int taille_vect = var->at(0).size();

	fwrite(&nb_vect,sizeof(unsigned int),1,f);
	fwrite(&taille_vect,sizeof(unsigned int),1,f);

	for(unsigned int i = 0 ; i < nb_vect; i++)
	{

		for(unsigned int j = 0 ; j < taille_vect; j++)
		{

			fwrite( &( var->at(i)[j] ),sizeof(var->at(i)[j]),1,f);

		}

	}

	fclose(f);

}


vector<vector<float>> lecture_fichier_bin2D(string nom_fichier)
{

	size_t readElts;

	FILE* f = fopen(nom_fichier.c_str(),"rb");

	if(f == NULL)
	{

		throw int(err6);

	}

	unsigned int nb_vect ;
	unsigned int taille_vect ;

	readElts = fread(&nb_vect,sizeof(nb_vect),1,f);
	readElts = fread(&taille_vect,sizeof(taille_vect),1,f);

	float temp;

	vector<float> vect_temp(taille_vect,0);

	vector<vector<float>> data;
	
	int i = 0;
	int j = 0;
	do
	{

		readElts = fread(&temp,sizeof(temp),1,f);

		if(readElts)
		{

			vect_temp[j] = temp;
			j++;
		}

		if(j == taille_vect)
		{

			j = 0;
			data.push_back(vect_temp);

		}

	}

	while(readElts);

	fclose(f);

	return data;

}


//Effectue un descripteur SIFT pour l'ensemble des images contenues dans le repertoire localise dans la varible chemin en entree
//Les autres parametres sont ceux presentes lors de la description de la fonction sift_descr(). L objet renvoye est un vecteur 3D,
//la premiere coordonnee du vecteur correspond aux images du repertoire localise dans la variable chemin. La seconde coordonnee correspond
//aux points cle de l image consideree et la troisieme correspond aux differents descripteurs du point cle (un par nombre d orientation du
//point cle)

vector<vector<vector<vl_sift_pix> > > ens_sift_descr(string chemin, int nb_octaves, int nb_niveaux, int octave_init ,
	double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region, double taille_fenetre )
{

		//Declaration des differents parametres
		Mat im;
		vl_sift_pix *image;
		vector<vector<vector<vl_sift_pix> > > images_descriptor;
		vector<vector<vl_sift_pix> > test2;
		int largeur_image;
		int hauteur_image;

		//Extraction du chemin de chaque fichier presents dans le repertoire localise dans la varible chemin
		vector<string> fichiers = ensemble_fichier(chemin.c_str());

		//Pour chaque fichier :
		for(unsigned int i = 0 ; i < fichiers.size() ; i++)
		{

			//Creation d un objet im a partir du fichier image consideree
			im = imread(fichiers[i], -1);

			//Conversion des element de l objet im en float
			im.convertTo(im,CV_32F);
			largeur_image = im.cols;
			hauteur_image = im.rows;

			//Convertit l image en pointeur float
			image = transf_point(&im);

			//Calcul du descripteur associe a l image consideree
			test2 =  sift_descr(image, largeur_image, hauteur_image, nb_octaves, nb_niveaux, octave_init , seuil_courbure_contours,
				seuil_extrema_locaux, taille_region, taille_fenetre );
			images_descriptor.push_back( test2 );
			free(image);

		}

		return images_descriptor;
}


//Meme fonction que precedemment a la difference pres qu il y a en entree un ensemble de repertoires, chacun contenant les images d une meme classe  
//et que le vecteur retourne est en 4 dimensions : il s agit d un ensemble de vecteurs 3D generes a l aide de la methode precedente : un pour 
//chaque repertoire contenu dans la variable chemin ou bien un par classe de defaut vu que les images de classe identique sont dans le meme repertoire

vector<  vector<vector<vector<vl_sift_pix> > >  > ens_sift_descr(vector<string> *chemin, int nb_octaves, int nb_niveaux, int octave_init ,
	double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region, double taille_fenetre )
{

		//Declaration des differents parametres
		Mat im;
		vl_sift_pix *image;
		vector<  vector<vector<vector<vl_sift_pix> > >  > res;
		vector<vector<vector<vl_sift_pix> > > images_descriptor;
		int largeur_image;
		int hauteur_image;
		vector<vector<string> > fichiers;

		//Extraction du chemin de chaque fichier presents dans lees repertoires localises dans la varible chemin
		for(unsigned int k =  0 ; k < chemin->size() ; k++)
		{

			fichiers.push_back(ensemble_fichier(chemin->at(k).c_str()));

		}

		//Pour chaque repertoire :
		for(unsigned int k =  0 ; k < fichiers.size() ; k++)
		{
			//Pour chaque fichier :
			for(unsigned int i = 0 ; i < fichiers[k].size() ; i++)
			{

				//Creation d un objet im a partir du fichier image consideree
				im = imread(fichiers[k][i], -1);

				//Conversion des element de l objet im en float
				im.convertTo(im,CV_32F);
				largeur_image = im.cols;
				hauteur_image = im.rows;

				//Convertit l image en pointeur float
				image = transf_point(&im);

				//Calcul du descripteur associe a l image consideree
				images_descriptor.push_back( sift_descr(image, largeur_image, hauteur_image, nb_octaves, nb_niveaux, octave_init , seuil_courbure_contours,
					seuil_extrema_locaux, taille_region, taille_fenetre ) );
				free(image);

			}

			res.push_back(images_descriptor);
			images_descriptor.clear();

		}

		return res;
}


//Prend en entree les chemins d un ensemble de repertoires, chacun contenant les images d une meme classe  et retourne leurs descripteurs SIFT + FV, ainsi que
//les parametres permettant de generer l objet GMM utilise. Les autres parametres sont les memes que precedemment

vector<vector<float>> ens_sift_fv_descr(vector<string> *chemin, int nb_octaves, int nb_niveaux, int octave_init,
	int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region, double taille_fenetre,
	int kmean_max_nb_iter, int kmean_nb_repet, string bin1, string bin2, string bin3)
{
		//On commence par determiner l ensemble des descripteurs SIFT
		vector< vector<vector<vector<vl_sift_pix> > > > descr_tout_images =  ens_sift_descr(chemin, nb_octaves, nb_niveaux, octave_init,
			seuil_courbure_contours, seuil_extrema_locaux, taille_region, taille_fenetre );

		//declaration et initialisation de varaibles
		int taille_descr_pt = 0;
		int nb_images;
		int nb_descr = 0;
		vector<vector<float> > ens_data;
		vector<float> temp;
		int nb_float = 0;
		vl_sift_pix *data;
		vector<vl_sift_pix*> data_separee;
		int k , m, o;
		VlGMM* gmm;
		void *enc;
		float *fishervector;

		//Pour chaque repertoire en entree (ou pour chaque classe de defaut)
		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			//On determine le nombre d images par repertoire/classe
			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			//Pour chaque image 
			for(int l = 0 ; l <nb_images; l++)
			{

				//Mise a jour du nombre total de float decrivant l ensemble de nos images
				nb_float += descr_tout_images[n][l].size()*taille_descr_pt;

				//Mise a jour du nombre total de descripteurs pour  l ensemble de nos images
				nb_descr += descr_tout_images[n][l].size();

				//Allocation d espace memoire pour chaque element du vecteur data_separee qui contiendra chaque descripteur de point cle 
				//regroupe par image : 
				data_separee.push_back( (float*)calloc(descr_tout_images[n][l].size()*taille_descr_pt,sizeof(float)) );

			}

		}

		//Allocation d espace memoire pour le vecteur data qui contiendra  lensemble des descripteur de point cle
		data =  (float*)calloc(nb_float,sizeof(float));

		k = 0;
		m = 0;
		o = 0;

		//Pour chaque repertoire en entree (ou pour chaque classe de defaut)
		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{
			//On determine le nombre d images par repertoire/classe
			taille_descr_pt = descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			//Pour chaque image 
			for(int l = 0 ; l < nb_images; l++)
			{

				//Pour chaque descripteur de point cle 
				for(unsigned int i = 0 ; i < descr_tout_images[n][l].size(); i++)
				{

					//Pour chaque valeur du descripteur 
					for( int j = 0 ; j < taille_descr_pt; j++)
					{

						//On rempit lesvarables data et data_separee
						data[k] =  descr_tout_images[n][l][i][j];
						data_separee[o][m] = descr_tout_images[n][l][i][j];
						k++;
						m++;

					}

				}

				m = 0;
				o++;
			}

		}

		//Creation de l objet GMM qui va nous permettre de reduire la taille des descripteurs
		gmm = vl_gmm_new (VL_TYPE_FLOAT,taille_descr_pt, nb_gauss);

		//Creation de l objet Kmean qui vas nous permettre d initialiser certains parametres de l objet GMM
		VlKMeans *kmoy = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);

		//Initialisation du Kmean avec la methode VlKMeansPlusPlus (voir la doc de vlfeat pour plus de details)
		vl_kmeans_set_initialization(kmoy,VlKMeansPlusPlus);

		//Choix de l algorithme Elkan (voir doc pour plus de details) pour notre kmean
		vl_kmeans_set_algorithm(kmoy,VlKMeansElkan);

		//On precise avec quelles donnees , leur dimension , leur nombre et le nombre de cluster on souhaite determiner notre Kmean
		vl_kmeans_init_centers_plus_plus(kmoy, data, taille_descr_pt, nb_descr, nb_gauss);

		//On fixe le nombre maximum d iteration pour l algo Elkan 
		vl_kmeans_set_max_num_iterations (kmoy, kmean_max_nb_iter);

		//Permet d obtenir des centres plus precis
		vl_kmeans_refine_centers (kmoy, data, nb_descr);

		//On fixe le nombre maximum de repetition de l algo Elkan 
		vl_kmeans_set_num_repetitions(kmoy,kmean_nb_repet);

		//Effectue l algo Elkan pour determiner les parametres a utiliser pour l objet GMM
		vl_kmeans_cluster(kmoy , data , taille_descr_pt , nb_descr , nb_gauss);

		//Choix de l algorithme utilise pour l initialisation des donnee ici Kmean
		vl_gmm_set_initialization (gmm,VlGMMKMeans);

		//On precise l objet de type Kmean utilise pour initialiser les parametres
		vl_gmm_set_kmeans_init_object( gmm , kmoy );

		//Applique la methode GMM aux donnees
		vl_gmm_cluster(gmm, data, nb_descr);

		//Allocation d espace memoire a la variable enc qui contiendra les vecteurs de Fisher qui composerons les descripteurs des images
		enc =  vl_malloc(sizeof(float)*2*taille_descr_pt*nb_gauss);

		//Recuperation des 3 parametres qui nous permettront de reutiliser le meme objet GMM sans avoir a repasser par le Kmean
		void const* moy = vl_gmm_get_means(gmm);
		void const* cov = vl_gmm_get_covariances(gmm);
		void const* apriori = vl_gmm_get_priors(gmm);

		//Cast en double des parametres precedents
		double *moy2 = (double*)(moy);
		double *cov2 = (double*)(cov);
		double *apriori2 = (double*)(apriori);

		//"Cast" des parametres precedents en vecteur + sauvegarde dans des fichiers binaires
		vector<double> moy_vect =  point_en_vect(moy2, nb_gauss*64);
		ecriture_fichier_bin_double(&moy_vect, bin1);

		vector<double> cov_vect =  point_en_vect(cov2, nb_gauss*64);
		ecriture_fichier_bin_double(&cov_vect, bin2);

		vector<double> apriori_vect =  point_en_vect(apriori2, nb_gauss);
		ecriture_fichier_bin_double(&apriori_vect, bin3);

		o = 0;
		//Pour chaque classe/repertoire
		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			//On determine le nombre d images par repertoire/classe
			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();
			//Pour chaque image
			for(int l = 0 ; l < nb_images; l++)
			{
				//Creation des vecteurs de Fisher pour chaque image qui sont stockes dans enc
				vl_fisher_encode(
					enc, VL_TYPE_FLOAT,
					moy2, taille_descr_pt, nb_gauss,
					cov2,
					apriori2,
					data_separee[o], descr_tout_images[n][l].size(),
					VL_FISHER_FLAG_IMPROVED
					);
				//Cette etape est tres importante : nous sommes passes d images decrites par un ensemble de point cle eux meme decrit par plusieurs vecteurs
				//a une image decrite par un unique vecteur de taille FIXE ce qui est beaucoup plus simple pour la classification

				//cast du vecteur de Fisher en float
				fishervector = (float*)(enc);

				//Sauvegarde du vecteur de Fisher 
				temp = point_en_vect(fishervector, 2*taille_descr_pt*nb_gauss);
				ens_data.push_back(temp);
				temp.clear();
				o++;

			}

		}

		free(data);
		vl_free(enc);

		for(unsigned int i = 0 ; i < data_separee.size(); i++)
		{

			free(data_separee[i]);

		}


		return ens_data;

}


//Prend en entree les chemins d un ensemble de repertoires, chacun contenant les images d une meme classe et retourne leurs descripteurs SIFT + FV, ainsi que
//l objet GMM utilise. Les autres parametres sont les memes que precedemment a lexception des 3 derniers qui permettront d obtenir un objet GMM
//sans passer par l algo Kmean 

vector<vector<float> > ens_sift_fv_descr(vector<string> *chemin, int nb_octaves, int nb_niveaux,
	int octave_init, int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux,
	double taille_region, double taille_fenetre, vector<double> *moy, vector<double> *cov, vector<double> *apriori)
{
		//On commence par determiner l ensemble des descripteurs SIFT
		vector< vector<vector<vector<vl_sift_pix> > > > descr_tout_images =  ens_sift_descr(chemin, nb_octaves, nb_niveaux, octave_init,
			seuil_courbure_contours, seuil_extrema_locaux, taille_region, taille_fenetre );

		//declaration et initialisation de varaibles
		int taille_descr_pt = 0;
		int nb_images;
		int nb_descr = 0;
		vector<vector<float> > ens_data;
		vector<float> temp;
		int nb_float = 0;
		vl_sift_pix *data;
		vector<vl_sift_pix*> data_separee;
		int k , m, o;
		void *enc;
		float *fishervector;

		//Pour chaque repertoire en entree (ou pour chaque classe de defaut)
		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			//On determine le nombre d images par repertoire/classe
			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			//Pour chaque image 
			for(int l = 0 ; l <nb_images; l++)
			{
				//Mise a jour du nombre total de float decrivant l ensemble de nos images
				nb_float += descr_tout_images[n][l].size()*taille_descr_pt;

				//Mise a jour du nombre total de descripteurs pour  l ensemble de nos images
				nb_descr += descr_tout_images[n][l].size();

				//Allocation d espace memoire pour chaque element du vecteur data_separee qui contiendra chaque descripteur de point cle 
				//regroupe par image : 
				data_separee.push_back( (float*)calloc(descr_tout_images[n][l].size()*taille_descr_pt,sizeof(float)) );

			}
		}

		//Allocation d espace memoire pour le vecteur data qui contiendra  lensemble des descripteur de point cle
		data =  (float*)calloc(nb_float,sizeof(float));

		k = 0;
		m = 0;
		o = 0;


		//Pour chaque repertoire en entree (ou pour chaque classe de defaut)
		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{
			//On determine le nombre d images par repertoire/classe
			taille_descr_pt = descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			//Pour chaque image 
			for(int l = 0 ; l < nb_images; l++)
			{

				//Pour chaque descripteur de point cle 
				for(unsigned int i = 0 ; i < descr_tout_images[n][l].size(); i++)
				{

					//Pour chaque valeur du descripteur 
					for( int j = 0 ; j < taille_descr_pt; j++)
					{

						//On rempit les varables data et data_separee
						data[k] =  descr_tout_images[n][l][i][j];
						data_separee[o][m] = descr_tout_images[n][l][i][j];
						k++;
						m++;

					}

				}

				m = 0;
				o++;
			}

		}

		//Allocation d espace memoire a la variable enc qui contiendra les vecteurs de Fisher qui composerons les descripteurs des images
		enc =  vl_malloc(sizeof(float)*2*taille_descr_pt*nb_gauss);

		o = 0;

		//conversion des vecteur en entree en pointeurs 
		double* moy_point = vect_en_point(moy);
		double* cov_point = vect_en_point(cov);
		double* apriori_point = vect_en_point(apriori);

		//Pour chaque repertoire en entree (ou pour chaque classe de defaut)
		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			//On determine le nombre d images par repertoire/classe
			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			//Pour chaque image
			for(int l = 0 ; l < nb_images; l++)
			{

				//Creation des vecteurs de Fisher pour chaque image qui sont stockes dans enc
				vl_fisher_encode(
					enc, VL_TYPE_FLOAT,
					moy_point, taille_descr_pt, nb_gauss,
					cov_point,
					apriori_point,
					data_separee[o], descr_tout_images[n][l].size(),
					VL_FISHER_FLAG_IMPROVED
					);

				//cast du vecteur de Fisher en float
				fishervector = (float*)(enc);

				//Sauvegarde du vecteur de Fisher
				temp = point_en_vect(fishervector, 2*taille_descr_pt*nb_gauss);
				ens_data.push_back(temp);
				temp.clear();
				o++;

			}

		}

		free(data);
		vl_free(enc);

		for(unsigned int i = 0 ; i < data_separee.size(); i++)
		{

			free(data_separee[i]);

		}

		return ens_data;

}


//Prend en entree le chemin d'un unique repertoire contenant un ensemble d images et retourne leurs descripteurs SIFT + FV. 
//Les autres parametres sont les memes que precedemment

vector<vector<float> > ens_sift_fv_descr(string chemin, int nb_octaves, int nb_niveaux, int octave_init, int nb_gauss,
	double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region, double taille_fenetre,
	vector<double> *moy, vector<double> *cov, vector<double> *apriori)
{
		//On commence par determiner l ensemble des descripteurs SIFT
		vector<vector<vector<vl_sift_pix> > > descr_tout_images = ens_sift_descr(chemin, nb_octaves, nb_niveaux, octave_init,
			seuil_courbure_contours, seuil_extrema_locaux, taille_region, taille_fenetre );

		//declaration et initialisation de varaibles
		int taille_descr_pt = descr_tout_images[0][0].size();
		int nb_descr = 0;
		vector<vector<float> > ens_data;
		vector<float> temp;
		int nb_float = 0;
		vector<vl_sift_pix*> data_separee;
		int m;
		void *enc;
		float *fishervector;

		//Pour chaque image en entree (ou pour chaque classe de defaut)
		for(unsigned int l = 0 ; l < descr_tout_images.size(); l++)
		{
			//On determine le nombre de descripteur ainsi que le nombre de variables composant l ensemble des descripteurs 
			nb_float += descr_tout_images[l].size()*taille_descr_pt;
			nb_descr += descr_tout_images[l].size();

			//Allocation d espace memoire pour chaque element du vecteur data_separee qui contiendra chaque descripteur de point cle 
			//regroupe par image : 
			data_separee.push_back( (float*)calloc(descr_tout_images[l].size()*taille_descr_pt,sizeof(float)) );

		}

		m = 0;

		//Pour chaque image en entree 
		for(unsigned int l = 0 ; l < descr_tout_images.size(); l++)
		{

			//Pour chaque descripteur 
			for(unsigned int i = 0 ; i < descr_tout_images[l].size(); i++)
			{

				//Pour chaque composant de descripteur
				for( int j = 0 ; j < taille_descr_pt; j++)
				{
					//On rempit la varable data_separee
					data_separee[l][m] = descr_tout_images[l][i][j];
					m++;

				}

			}

			m = 0;

		}

		//Allocation d espace memoire a la variable enc qui contiendra les vecteurs de Fisher qui composerons les descripteurs des imagess
		enc =  vl_malloc(sizeof(float)*2*taille_descr_pt*nb_gauss);

		//conversion des vecteur en entree en pointeurs
		double* moy_point = vect_en_point(moy);
		double* cov_point = vect_en_point(cov);
		double* apriori_point = vect_en_point(apriori);

		//Pour chaque image en entree 
		for(unsigned int l = 0 ; l < descr_tout_images.size(); l++)
		{

			//Creation des vecteurs de Fisher pour chaque image qui sont stockes dans enc
			vl_fisher_encode(
				enc, VL_TYPE_FLOAT,
				moy_point, taille_descr_pt, nb_gauss,
				cov_point,
				apriori_point,
				data_separee[l], descr_tout_images[l].size(),
				VL_FISHER_FLAG_IMPROVED
				);

			//cast du vecteur de Fisher en float
			fishervector = (float*)(enc);

			//Sauvegarde du vecteur de Fisher
			temp = point_en_vect(fishervector, 2* taille_descr_pt *nb_gauss);
			ens_data.push_back(temp);
			temp.clear();

		}

		vl_free(enc);

		for(unsigned int i = 0 ; i < data_separee.size(); i++)
		{

			free(data_separee[i]);

		}

		free(moy_point);
		free(cov_point);
		free(apriori_point);

		return ens_data;

}


//Meme fonction que precedemment a la difference pres que la methode recoit un ensemble de repertoires, chacun contenant les images d une meme classe

vector<vector<float> > ens_sift_fv_descr1(vector<string> *chemin, int nb_octaves, int nb_niveaux, int octave_init,
	int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region, double taille_fenetre,
	vector<double> *moy, vector<double> *cov, vector<double> *apriori)
{
	
		vector< vector<vector<vector<vl_sift_pix> > > > descr_tout_images =  ens_sift_descr(chemin, nb_octaves,
			nb_niveaux, octave_init, seuil_courbure_contours, seuil_extrema_locaux, taille_region, taille_fenetre );

		int taille_descr_pt = 0;
		int nb_images;
		int nb_descr = 0;
		vector<vector<float> > ens_data;
		vector<float> temp;
		int nb_float = 0;
		vl_sift_pix *data;
		vector<vl_sift_pix*> data_separee;
		int k , m;
		void *enc;
		float *fishervector;

		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			for(int l = 0 ; l <nb_images; l++)
			{

				nb_float += descr_tout_images[n][l].size()*taille_descr_pt;
				nb_descr += descr_tout_images[n][l].size();
				data_separee.push_back( (float*)calloc(descr_tout_images[n][l].size()*taille_descr_pt,sizeof(float)) );

			}
		}

		data =  (float*)calloc(nb_float,sizeof(float));
		k = 0;
		m = 0;

		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			for(int l = 0 ; l < nb_images; l++)
			{

				for(unsigned int i = 0 ; i < descr_tout_images[n][l].size(); i++)
				{

					for( int j = 0 ; j < taille_descr_pt; j++)
					{

						data[k] =  descr_tout_images[n][l][i][j];
						data_separee[l][m] = descr_tout_images[n][l][i][j];
						k++;
						m++;

					}

				}

				m = 0;

			}

		}

		enc =  vl_malloc(sizeof(float)*2*taille_descr_pt*nb_gauss);

		double* moy_point = vect_en_point(moy);
		double* cov_point = vect_en_point(cov);
		double* apriori_point = vect_en_point(apriori);

		for(unsigned int n = 0 ; n < descr_tout_images.size() ; n++ )
		{

			taille_descr_pt =  descr_tout_images[n][0][0].size();
			nb_images = descr_tout_images[n].size();

			for(int l = 0 ; l < nb_images; l++)
			{

				vl_fisher_encode(
					enc, VL_TYPE_FLOAT,
					moy_point, taille_descr_pt, nb_gauss,
					cov_point,
					apriori_point,
					data_separee[l], descr_tout_images[n][l].size(),
					VL_FISHER_FLAG_IMPROVED
					);

				fishervector = (float*)(enc);
				temp = point_en_vect(fishervector, 2*taille_descr_pt*nb_gauss);
				ens_data.push_back(temp);
				temp.clear();

			}

		}

		free(data);
		vl_free(enc);

		for(unsigned int i = 0 ; i < data_separee.size(); i++)
		{

			free(data_separee[i]);

		}

		return ens_data;

}


//Retourne l objet SVM entraine sur les descripteurs et leurs etiquettes associes en entree

//Le vecteur etiquettes contient les labels associes aux images dont les descripteurs sont en entree les seuls valeurs acceptes sont 1 et -1 (pour des
//raisons mysterieuses cette variable doit obligatoirement etre de type double*) si l image a un label egal a 1 elle est definie comme positive : elle
//appartient a la classe consideree sinon elle est definie comme negative : elle n appartient pas a la  classe consideree : ici, le SVM est un classifieur
//binaire : uniquement 2 classes

//Le parametre lambda (strictement positif!!!!!!!!) permet de donner plus ou moins d importance a la norme de w dans le probleme d optimisation que le svm
//doit resoudre (plus de details dans le rapport). Une valeur faible ( 10^-5 ) donnera plus d importance a la minimisation de l erreur lors de la phase
//d apprentissage : le modele sera tres precis pour classer les elements qu'il connait deja mais aura du mal a se generaliser a de nouvelles donnees
//(probleme de surapprentissage). A l inverse une valeur elevee (0.1) de lambda diminuera le risque de surapprentissage mais le modele ne sera peut etre
//pas assez specifique et ne sera meme pas capable de classer (avec une precision satisfaisante) les elements de la base d apprentissage

VlSvm * model_svm(vector<vector<float> > *descripteur , double *etiquettes, double lambda)
{

	//Declaration et initialisation de variables
	vl_size const numData = descripteur->size();
	vl_size const dimension = descripteur->at(0).size();

	float *data = vect2_en_point(descripteur);
	double *data_conv = (double*)vl_malloc(sizeof(double)*numData*dimension);

	//Conversion de la variable descripteur en entree en pointeur sur double
	for(unsigned int i = 0 ; i < numData*dimension ; i++ )
	{

		data_conv[i] = (double)(data[i]);

	}

	//Creation de l objet SVM a l aide des differentes variables en entree et definies precedemment. On precise egalement l algorithme qui sera utilise
	//lors de la phased apprentissage : ici, on utilise l algo SDCA (plus de details dans le rapport)
	VlSvm * svm = vl_svm_new(VlSvmSolverSdca,
		data_conv,
		dimension,
		numData,
		etiquettes,
		lambda);

	//Phase d apprentissage
	vl_svm_train(svm);

	vl_free(data_conv);
	free(data);

	return svm;
}



//Retourne le produit scalaire entre le pointeur et le vecteur en entree (doivent etre de meme taille) auxquels on rajoute un terme supplementaire

double prod_scal_svm(const double *model , double biais , vector<float> *descr_fv , double B){

	double score = 0;

	for(unsigned int i = 0 ; i < (*descr_fv).size() ; i++){

		score+=  model[i]*(*descr_fv)[i];

	}

	score+=B*biais;

	return score;

}




//Retourne le score (associe au SVM en entree) du descripteur en entree en effectuant un produit scalaire entre le vecteur w du modele svm
//et le descripteur en entree : si le resultat est positif l image associee au descripteur en entree appartient a la classe positive,
//sinon elle n y appartient pas

double score_svm(VlSvm *svm , vector<float> *descr_fv)
{

	//Obtention du vecteur w
	const double *model = vl_svm_get_model(svm);

	//Obtention du biais
	double bias = vl_svm_get_bias(svm);

	//Obtention du multiplicateur de biais
	double biais_multiplier = vl_svm_get_bias_multiplier(svm);

	//Realisation du produit sclaaire  entre w et descr_fv en ajoutant le biais comme dernier element de w et le multiplicateur de biais comme dernier
	//element de descr_fv
	double score =  prod_scal_svm(model , bias , descr_fv , biais_multiplier);

	return score;

}



//Retourne l ensemble des scores (associe au SVM en entree) des descripteurs en entree

vector<double> ens_score_svm(VlSvm *svm , vector<vector<float> > *descr_fv)
{

	vector<double> scores;

	for(unsigned  int i = 0 ; i < (*descr_fv).size() ; i++)
	{

		scores.push_back( score_svm(svm ,  &((*descr_fv)[i]) ) );

	}

	return scores;

}




//Retourne l ensemble des scores (associe au SVM en entree) de l ensemble des descripteur associes aux images contenues dans le repertoire en entree
//determines a l aide du modele GMM en entree. Les autres parametres sont les meme que precedemment

vector<double> ens_score_svm(VlSvm * svm ,string chemin, int nb_octaves, int nb_niveaux, int octave_init, int nb_gauss,
	double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region, double taille_fenetre, vector<double> *moy,
	vector<double> *cov, vector<double> *apriori)
{

		vector<vector<float> > descr = ens_sift_fv_descr(chemin, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours,
			seuil_extrema_locaux, taille_region, taille_fenetre,moy, cov, apriori);

		vector<double> scores = ens_score_svm(svm , &descr);

		return scores;
}



//Meme fonction que precedemment a la difference pres que les images de la base d'apprentissage sont prises dans un ensemble de
//repertoires dont les chemins sont precise dans le vector chemin_entrainement

vector<double> score_test(vector<vector<float> > *sift_fv_descr_appr, vector<vector<float> > *sift_fv_descr_test,
	double* etiquettes, double lambda)
{

		VlSvm * svm;

		svm = model_svm(sift_fv_descr_appr , etiquettes, lambda);

		vector<double> scores_appr = ens_score_svm(svm , sift_fv_descr_appr);

		int nb_pos = 0;
		int nb_neg = 0;
		int err_pos = 0;
		int err_neg = 0;

		for( unsigned int i = 0 ; i < scores_appr.size() ; i++ )
		{

			if(etiquettes[i] < 0)
			{

				nb_neg++;

			}

			if(etiquettes[i] > 0)
			{

				nb_pos++;

			}

			if( scores_appr[i] > 0 && etiquettes[i] < 0 )
			{

				cout<< "Erreur negative image "<< i <<endl;
				err_neg++;

			}

			if( scores_appr[i] < 0 && etiquettes[i] > 0 )
			{

				cout<< "Erreur positive image "<< i <<endl;
				err_pos++;

			}

		}

		cout<<"Test sur la base d apprentissage : "<<endl;
		cout<<"Nombre d erreurs positives : "<< err_pos << " sur "<<nb_pos<<"  echantillons positifs "<<endl;
		cout<<"Nombre d erreurs negatives : "<< err_neg << " sur "<<nb_neg<<"  echantillons negatifs "<<endl;

		vector<double> scores_test = ens_score_svm(svm , sift_fv_descr_test);

		vl_svm_delete(svm);

		return scores_test;
}


//Meme fonction que precedemment a l exception de la variable chemin_entrainement qui cette fois est un ensemble de chemins vers les repertoires
//contenant chacun les images appartenant a une meme classe

vector<int> classement_one_vs_all(vector<vector<float> > *sift_fv_descr_appr, vector<vector<float> > *sift_fv_descr_test,
	vector<double*> etiquettes, double lambda)
{


		vector<vector<double> > ens_scores;

		unsigned int nb_classes =  etiquettes.size();

		clock_t begin = clock();

		for(unsigned int i = 0 ; i <  nb_classes ; i++)
		{

			cout<<"Classe "<<i+1<<endl;
			ens_scores.push_back(score_test(sift_fv_descr_appr, sift_fv_descr_test, etiquettes[i], lambda));
			cout<<"Fin classe "<<i+1<<endl;
			cout<<endl;
		}

		unsigned int nb_obs = ens_scores[0].size();

		int classe;
		double max;

		vector<int> classes;

		for(unsigned int i = 0 ; i < nb_obs; i++)
		{

			max = INT_MIN;
			classe = 0;

			for(unsigned j = 0 ; j < nb_classes ; j++)
			{

				if(ens_scores[j][i] > max)
				{

					max = ens_scores[j][i];
					classe = j+1;

				}

			}
			//cout<< "Score de l image "<< i+1 <<" : "<< max <<endl;
			classes.push_back(classe);

		}

		clock_t end = clock();

		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		cout<<"Temps total classement : "<< elapsed_secs <<endl;

		for(unsigned int i = 0 ; i < nb_classes ; i++)
		{

			vl_free(etiquettes[i]);

		}

		return classes;

}



//Meme fonction que precedemment a l exception de l'abscence de la variable etiquette qui est determine de facon automatique

vector<int> classement_one_vs_all_etiquet_auto(vector<string> *chemin_entrainement, string chemin_test, int nb_octaves, int nb_niveaux,
	int octave_init, int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region,
	double taille_fenetre, int kmean_max_nb_iter, int kmean_nb_repet, double lambda, string bin1, string bin2, string bin3)
{

		vector<double*> etiquettes;
		double *temp;
		vector<double> moy;
		vector<double> cov;
		vector<double> apriori;

		int nb_images_entrainement = 0;

		//Calcul de nombre total d images dans la base d apprentissage
		for(unsigned int i = 0 ; i < chemin_entrainement->size() ; i++)
		{

			nb_images_entrainement += nombre_fichier(chemin_entrainement->at(i).c_str());

		}

		int position1 = 0;
		int position2 = 0;

		//Generation de la variable etiquettes a l aide de la variable chemin_entrainement
		for(unsigned int i = 0 ; i < chemin_entrainement->size() ; i++)
		{

			position2 += nombre_fichier(chemin_entrainement->at(i).c_str());

			temp = (double*)vl_malloc(sizeof(double)*nb_images_entrainement);

			for(int j = 0 ; j < nb_images_entrainement ; j++)
			{

				if(j < position1 || j >= position2)
				{

					temp[j] = -1;

				}

				else
				{

					temp[j] = 1;

				}

			}

			position1 += nombre_fichier(chemin_entrainement->at(i).c_str());
			etiquettes.push_back(temp);
		}

		vector<vector<float> > sift_fv_descr_appr;
		vector<vector<float> > sift_fv_descr_test;

		clock_t begin = clock();

		cout<<"Debut creation descripteurs"<<endl;

		sift_fv_descr_appr = ens_sift_fv_descr(chemin_entrainement, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux,
			taille_region, taille_fenetre, kmean_max_nb_iter, kmean_nb_repet, bin1, bin2, bin3);

		vector<double> moy2 = lecture_fichier_bin_double("moy.bin");
		vector<double> cov2 = lecture_fichier_bin_double("cov.bin");
		vector<double> apriori2 = lecture_fichier_bin_double("apriori.bin");

		cout<<"Descripteurs apprentissage termines "<<endl;

		sift_fv_descr_test = ens_sift_fv_descr(chemin_test, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux,
			taille_region,  taille_fenetre, &moy2, &cov2, &apriori2);

		cout<<"Fin creation descripteurs"<<endl;

		clock_t end = clock();

		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		cout<<"Temps total creation descripteurs : "<< elapsed_secs <<endl;
		cout<<endl;
		cout<<endl;

		return classement_one_vs_all( &sift_fv_descr_appr, &sift_fv_descr_test, etiquettes, lambda);

}


//Prend en entree le vecteur d'entiers contenant les classes des images détermines par le SVM et le compare avec le vecteur
//d'entiers contenant les bonnes classes des images.

void test_resultats(vector<int> *classes, vector<int> *etiquettes_test, int nb_classes )
{

	vector<int> compteur_err(nb_classes, 0);
	vector<int> compteur(nb_classes, 0);
	int compteur_total = 0;

	for(unsigned int i = 0 ; i < classes->size() ; i++)
	{

		for(int j = 0 ; j < nb_classes ; j++)
		{

			if(etiquettes_test->at(i)==j+1)
			{

				compteur[j]++;

			}

			if( ( classes->at(i) != etiquettes_test->at(i) ) && (etiquettes_test->at(i) == j+1) )
			{

				compteur_total++;
				compteur_err[j]++;
			}

		}

	}

	cout<<endl;

	for(int i = 0 ; i < nb_classes ; i++)
	{

		cout<<"Nombre d erreurs pour la classe "<< i+1 <<" : "<< compteur_err[i] <<" sur "<< compteur[i] <<endl;
		cout<<"Soit une precision de : "<< 100*( 1 - (double)compteur_err[i]/(double)compteur[i] ) <<"%"<<endl;
		cout<<endl;

	}

	cout<<"Nombre d erreurs totale : "<< compteur_total <<endl;
	cout<<"Soit une precision de : "<< 100*( 1 - (double)compteur_total/(double)classes->size() ) <<"%"<<endl;

}



//Cree un descripteur a partir de la base d'apprentissage et des parametres en entree et sauvegarde  dans des fichiers binaires les
//informations necessaires a la creation du meme descripteur par la suite

vector< vector<float> > creation_sauvegarde_descr(vector<string> *chemin_entrainement, int nb_octaves, int nb_niveaux,
	int octave_init, int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region,
	double taille_fenetre, int kmean_max_nb_iter, int kmean_nb_repet, string bin1, string bin2, string bin3)
{


		vector<vector<float> > sift_fv_descr_appr;

		cout<<"Debut creation descripteurs"<<endl;

		sift_fv_descr_appr = ens_sift_fv_descr(chemin_entrainement, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux,
			taille_region, taille_fenetre, kmean_max_nb_iter, kmean_nb_repet, bin1, bin2, bin3);

		cout<<"Creation descripteurs termine"<<endl;

		return sift_fv_descr_appr;

}



//Classe les elements en entree en les generant a l aide des informations contenues dans les fichiers bin en entree

vector<int> classement(vector<string> *chemin_entrainement, string chemin_test, int nb_octaves, int nb_niveaux,
	int octave_init, int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region,
	double taille_fenetre, double lambda, string bin1, string bin2, string bin3,
	vector<vector<float>> sift_fv_descr_appr)
{


		vector<double*> etiquettes;
		double *temp;
		int nb_images_entrainement = 0;


		//Calcul de nombre total d images dans la base d apprentissage
		for(unsigned int i = 0; i < chemin_entrainement->size(); i++)
		{

			nb_images_entrainement += nombre_fichier(chemin_entrainement->at(i).c_str());

		}

		int position1 = 0;
		int position2 = 0;


		//Generation de la variable etiquettes a l aide de la variable chemin_entrainement
		for(unsigned int i = 0 ; i < chemin_entrainement->size() ; i++)
		{

			position2 += nombre_fichier(chemin_entrainement->at(i).c_str());

			temp = (double*)vl_malloc(sizeof(double)*nb_images_entrainement);

			for(int j = 0 ; j < nb_images_entrainement ; j++)
			{

				if(j < position1 || j >= position2)
				{

					temp[j] = -1;

				}

				else
				{

					temp[j] = 1;

				}

			}

			position1 += nombre_fichier(chemin_entrainement->at(i).c_str());
			etiquettes.push_back(temp);
		}


		vector<vector<float> > sift_fv_descr_test;
		vector<double> moy2 = lecture_fichier_bin_double(bin1);
		vector<double> cov2 = lecture_fichier_bin_double(bin2);
		vector<double> apriori2 = lecture_fichier_bin_double(bin3);

		sift_fv_descr_test = ens_sift_fv_descr(chemin_test, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux,
			taille_region,  taille_fenetre, &moy2, &cov2, &apriori2);

		cout<<"Fin creation descripteurs"<<endl;

		return classement_one_vs_all( &sift_fv_descr_appr, &sift_fv_descr_test, etiquettes, lambda);

}



//Genere et sauvegarde la base d'apprentissage (sous forme de vecteurs) ainsi que certaines variables permettant d utiliser le meme objet lors de la 
//generation de futurs descripteurs.

int Teaching(vector<string> *chemin_entrainement, int nb_octaves, int nb_niveaux,
	int octave_init, int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region,
	double taille_fenetre, int kmean_max_nb_iter, int kmean_nb_repet, string bin_moy, string bin_cov, string bin_apriori, string descripteur_apprentissage)
{

	int res = RAS;

	vector<vector<float> > sift_fv_descr_appr;

	cout<<"Debut creation descripteurs"<<endl;


	try
	{
		sift_fv_descr_appr = ens_sift_fv_descr(chemin_entrainement, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux,
			taille_region, taille_fenetre, kmean_max_nb_iter, kmean_nb_repet, bin_moy, bin_cov, bin_apriori);
	}
	catch(int erreur)
	{
		res = erreur;
		cerr<<erreur<<endl;
	}

	cout<<"Creation descripteurs termine"<<endl;

	try
	{
		ecriture_fichier_bin2D(&sift_fv_descr_appr, descripteur_apprentissage);
	}
	catch(int erreur)
	{
		cerr<<erreur<<endl;
	}

	return res;
}



//Classe les elements en entree en les generant a l aide des informations contenues dans les fichiers bin en entree

int classifying(vector<string> *chemin_entrainement, string chemin_test, int nb_octaves, int nb_niveaux,
	int octave_init, int nb_gauss, double seuil_courbure_contours, double seuil_extrema_locaux, double taille_region,
	double taille_fenetre, double lambda, string bin_moy, string bin_cov, string bin_apriori, string descripteur_apprentissage, string res_classement)
{

		int res = RAS;
		vector<double*> etiquettes;
		double *temp;
		int nb_images_entrainement = 0;

		//Calcul de nombre total d images dans la base d apprentissage
		for(unsigned int i = 0; i < chemin_entrainement->size(); i++)
		{

			try
			{

				nb_images_entrainement += nombre_fichier(chemin_entrainement->at(i).c_str());

			}

			catch(int erreur)
			{
			
				res = erreur;
				cerr<<erreur<<endl;
			
			}

		}

		int position1 = 0;
		int position2 = 0;


		//Generation de la variable etiquettes a l aide de la variable chemin_entrainement
		for(unsigned int i = 0 ; i < chemin_entrainement->size() ; i++)
		{

			try
			{

				position2 += nombre_fichier(chemin_entrainement->at(i).c_str());

			}

			catch(int erreur)
			{
			
				res = erreur;
				cerr<<erreur<<endl;
			
			}
			
			temp = (double*)vl_malloc(sizeof(double)*nb_images_entrainement);

			for(int j = 0 ; j < nb_images_entrainement ; j++)
			{

				if(j < position1 || j >= position2)
				{

					temp[j] = -1;

				}

				else
				{

					temp[j] = 1;

				}

			}

			position1 += nombre_fichier(chemin_entrainement->at(i).c_str());
			etiquettes.push_back(temp);

		}


		vector<vector<float> > sift_fv_descr_test;
		vector<double> moy2;
		vector<double> cov2;
		vector<double> apriori2;
		try
		{
		
			moy2 = lecture_fichier_bin_double(bin_moy);
		
		}

		catch (int erreur) 
		{
		
			res = erreur;
			cerr<<erreur<<endl;

		}

		try
		{
		
			cov2 = lecture_fichier_bin_double(bin_cov);
		
		}

		catch (int erreur) 
		{
		
			res = erreur;
			cerr<<erreur<<endl;

		}

		try
		{
		
			apriori2 = lecture_fichier_bin_double(bin_apriori);
		
		}

		catch (int erreur) 
		{
		
			res = erreur;
			cerr<<erreur<<endl;

		}
		
		try
		{
			sift_fv_descr_test = ens_sift_fv_descr(chemin_test, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux,
				taille_region,  taille_fenetre, &moy2, &cov2, &apriori2);
		}

		catch(int erreur)
		{
			res = erreur;
			cerr<<erreur<<endl;
		}

		

		cout<<"Fin creation descripteurs"<<endl;

		vector<vector<float>> sift_fv_descr_appr = lecture_fichier_bin2D(descripteur_apprentissage);

		vector<int> classement = classement_one_vs_all( &sift_fv_descr_appr, &sift_fv_descr_test, etiquettes, lambda);

		ecriture_fichier_bin_int(&classement, res_classement);

		return res;

}


int main()
{

	int nb_octaves =  2;
	int nb_niveaux =  6;
	int octave_init = -1;
	int nb_gauss = 3;
	double seuil_courbure_contours = 10;
	double seuil_extrema_locaux = 0;
	double lambda = 0.005;
	double taille_region = 4;
	double taille_fenetre_gauss = 2;
	int kmean_max_nb_iter = 5;
	int kmean_nb_repet = 2;

	string bin_moy = "C:\\data\\test_apprentissage_supervisee_gui2\\moy.bin";
	string bin_cov = "C:\\data\\test_apprentissage_supervisee_gui2\\cov.bin";
	string bin_apriori = "C:\\data\\test_apprentissage_supervisee_gui2\\apriori.bin";
	string res_classement = "C:\\data\\test_apprentissage_supervisee_gui2\\res_classement.bin";

	string descripteur_apprentissage = "C:\\data\\test_apprentissage_supervisee_gui2\\descr_appr.bin";

	string chemin_test = "C:\\data\\test_apprentissage_supervisee_gui2\\test1_comet_GM_part_scratch";

	vector<string> chemin_test2;
	chemin_test2.push_back(chemin_test);

	vector<string> chemin_apprentissage;

	string temp = "C:\\data\\test_apprentissage_supervisee_gui2\\comet";
	chemin_apprentissage.push_back( temp );

	string temp2 = "C:\\data\\test_apprentissage_supervisee_gui2\\GM";
	chemin_apprentissage.push_back( temp2 );

	string temp4 = "C:\\data\\test_apprentissage_supervisee_gui2\\part";
	chemin_apprentissage.push_back( temp4 );

	string temp6 = "C:\\data\\test_apprentissage_supervisee_gui2\\scratch";
	chemin_apprentissage.push_back( temp6 );

	unsigned int nb_classes = chemin_apprentissage.size();

	vector<int> classes;
	vector<int> etiquettes_test(200, 0 );

	for(int i = 0; i < 50 ; i++)
	{

		etiquettes_test[i] = 1;

	}

	for(int i = 50; i < 100 ; i++)
	{

		etiquettes_test[i] = 2;

	}

	for(int i = 100; i < 150 ; i++)
	{

		etiquettes_test[i] = 3;

	}

	for(int i = 150; i < 200 ; i++)
	{

		etiquettes_test[i] = 4;

	}
	
	int erreur = Teaching(&chemin_apprentissage, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux, taille_region,
	 taille_fenetre_gauss, kmean_max_nb_iter, kmean_nb_repet,  bin_moy, bin_cov,  bin_apriori, descripteur_apprentissage);

	cout<<"Erreur : "<<erreur<<endl;

	erreur = classifying(&chemin_apprentissage, chemin_test, nb_octaves, nb_niveaux, octave_init, nb_gauss, seuil_courbure_contours, seuil_extrema_locaux, taille_region,
		taille_fenetre_gauss, lambda, bin_moy, bin_cov, bin_apriori, descripteur_apprentissage, res_classement);

	cout<<"Erreur : "<<erreur<<endl;

	classes = lecture_fichier_bin_int(res_classement);

	test_resultats(&classes, &etiquettes_test, nb_classes );
	
	string fin;
	cin>>fin;
	
	return 0;

}
