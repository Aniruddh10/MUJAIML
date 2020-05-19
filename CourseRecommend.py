import os
import time
import gc
import argparse

# data science imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# utils import
from fuzzywuzzy import fuzz


class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with
    KNN implmented by sklearn
    """
    def __init__(self, path_course, path_ratings1):
        """
        Recommender requires path to data: course data and ratings data
        Parameters
        ----------
        path_course: str, course data file path
        path_ratings1: str, ratings data file path
        """
        self.path_course = path_course
        self.path_ratings1 = path_ratings1
        self.course_rating_thres = 0
        self.user_rating1_thres = 0
        self.model = NearestNeighbors()

    def set_filter_params(self, course_rating_thres, user_rating1_thres):
        """
        set rating frequency threshold to filter less-known courses and
        less active users
        Parameters
        ----------
        course_rating_thres: int, minimum number of ratings received by users
        user_rating1_thres: int, minimum number of ratings of a course by a user
        """
        self.course_rating_thres = course_rating_thres
        self.user_rating1_thres = user_rating1_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        """
        set model params for sklearn.neighbors.NearestNeighbors
        Parameters
        ----------
        n_neighbors: int, optional (default = 5)
        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        metric: string or callable, default 'minkowski', or one of
            ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        n_jobs: int or None, optional (default=None)
        """
        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _prep_data(self):
        """
        prepare data for recommender
        1. course-user scipy sparse matrix
        2. hashmap of course to row index in course-user scipy sparse matrix
        """
        # read data
        df_course = pd.read_csv(
            os.path.join(self.path_course),
            usecols=['CourseID', 'CourseName'],
            dtype={'CourseID': 'float32', 'CourseName': 'str'},
            encoding= 'unicode_escape'
            )
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings1),
            usecols=['UsersReviewed', 'CourseID', 'Ratings'],
            dtype={'UsersReviewed': 'float32', 'CourseID': 'float32', 'Ratings': 'float32'},
            encoding='unicode_escape')
        # filter data
        df_course_cnt = pd.DataFrame(
            df_ratings.groupby('CourseID').size(),
            columns=['count'])
        popular_course = list(set(df_course_cnt.query('count >= @self.course_rating_thres').index))  # noqa
        course_filter = df_ratings.CourseID.isin(popular_course).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('CourseID').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating1_thres').index))  # noqa
        users_filter = df_ratings.UsersReviewed.isin(active_users).values

        df_ratings_filtered = df_ratings[course_filter & users_filter]

        # pivot and create course-user matrix
        course_user_mat = df_ratings_filtered.pivot(
            index='CourseID', columns='UsersReviewed', values='Ratings').fillna(0)
        # create mapper from coursename to index
        hashmap = {
            course: i for i, course in
            enumerate(list(df_course.set_index('CourseID').loc[course_user_mat.index].CourseName)) # noqa
        }
        # transform matrix to scipy sparse matrix
        course_user_mat_sparse = csr_matrix(course_user_mat.values)

        # clean up
        del df_course, df_course_cnt, df_users_cnt
        del df_ratings, df_ratings_filtered
        gc.collect()
        return course_user_mat_sparse, hashmap

    def _fuzzy_matching(self, hashmap, fav_course):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map course CourseName name to index of the course in data
        fav_course: str, name of user input course
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for CourseName, idx in hashmap.items():
            ratio = fuzz.ratio(CourseName.lower(), fav_course.lower())
            if ratio >= 60:
                match_tuple.append((CourseName, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def _inference(self, model, data, hashmap,
                   fav_course, n_recommendations):
        """
        return top n similar course recommendations based on user's input course
        Parameters
        ----------
        model: sklearn model, knn model
        data: course-user matrix
        hashmap: dict, map course CourseName name to index of the course in data
        fav_course: str, name of user input course
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar course recommendations
        """
        # fit
        model.fit(data)
        # get input course index
        print('You have input course:', fav_course)
        idx = self._fuzzy_matching(hashmap, fav_course)
        # inference
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # return recommendation (CourseId, distance)
        return raw_recommends

    def make_recommendations(self, fav_course, n_recommendations):
        """
        make top n course recommendations
        Parameters
        ----------
        fav_course: str, name of user input course
        n_recommendations: int, top n recommendations
        """
        # get data
        course_user_mat_sparse, hashmap = self._prep_data()
        # get recommendations
        raw_recommends = self._inference(
            self.model, course_user_mat_sparse, hashmap,
            fav_course, n_recommendations)
        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('Recommendations for {}:'.format(fav_course))
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Course Recommender",
        description="Run KNN Course Recommender")
    parser.add_argument('--path', nargs='?', default=r'C:\Users\hp\Desktop\Internship\Dataset',
                        help='input data path')
    parser.add_argument('--course_filename', nargs='?', default='course.csv',
                        help='provide courses filename')
    parser.add_argument('--ratings1_filename', nargs='?', default='ratings1.csv',
                        help='provide ratings filename')
    parser.add_argument('--course_name', nargs='?', default='',
                        help='provide the CourseName')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n course recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    course_filename = args.course_filename
    ratings1_filename = args.ratings1_filename
    course_name = args.course_name
    top_n = args.top_n
    # initial recommender system
    recommender = KnnRecommender(
        os.path.join(data_path, course_filename),
        os.path.join(data_path, ratings1_filename))
    # set params
    recommender.set_filter_params(50, 50)
    recommender.set_model_params(20, 'brute', 'cosine', -1)
    # make recommendations
    recommender.make_recommendations(course_name, top_n)