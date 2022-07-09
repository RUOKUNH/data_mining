#     crawler.py
#     coding: utf-8

'''
This is a crawler without multiprocess acceleration that outputs two .csv file named `Crawled Data` and `Genre List`, four .txt files named `Running Log`, `Genre List`, `Derived List` and `Failed List` and a filefolder storing crawled posters of movies named `Movie Poster` based on the primeval data bars in file `links.csv` and `movie.csv`.
'''


#     phase 0     Import needful modules.

import os
import csv
import time
import logging
import requests
import unicodedata
from bs4 import BeautifulSoup
# from multiprocessing import Pool


#     phase 1     Declare all global variables and names of file related.

#   files
source_csv = 'links.csv'
refer_csv = 'movies.csv'
export_csv = 'Crawled Data.csv'
genre_list_csv = 'Genre List.csv'
log_txt = 'Running Log.txt'
genre_list_txt = 'Genre List.txt'
derived_list_txt = 'Derived List.txt'
failed_list_txt = 'Failed List.txt'
poster_folder = './Movie Poster/'

#   variables
mvid_list = []
imdbid_list = []
tmdbid_list = []
mvname_list = []
releaseyear_list = []
derived_list = []
genre_list = []
genre_set = set()

#   main page targeted for crawling
url = 'https://www.imdb.com/title/'
#   a needful link string suffix in crawling poster images
image_url_suffix = '/mediaindex?ref_=tt_ov_mi_sm'
#   This is the headers for module `requests` in crawling, which could be omitted.
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36 Edg/96.0.1054.34'}
#   This is the time set for launch the targeted pages and can be distinguished by different kinds of page.
launch_time = 3
# launch_time = [3, 3, 3]
#   Set a certain movie ID for debugging and this is mostly for encoding error.
# debugid = 1107


#     phase 2     Derive pre-knowledge from primeval datasheet and do pre-writing.

#   Open the sourse .csv file to read different kinds of ID of movies.
#   Note that all these IDs are not consistent.
with open(source_csv, newline = '') as f:
    reader = csv.reader(f)
    for col in reader:
        mvid_list.append(col[0])
        imdbid_list.append(col[1])
        tmdbid_list.append(col[2])

#   Read the names of movies.
#   Note that here reset the arguments of `encoding` and `errors` to prevent some mistakes in text, which is a must.
with open(refer_csv, encoding = 'gb18030', errors = 'ignore', newline = '') as f:
    reader = csv.reader(f)
    for col in reader:
        mvname_list.append(col[1])

#   Pop to discard the titles.
mvid_list.pop(0)
imdbid_list.pop(0)
tmdbid_list.pop(0)
mvname_list.pop(0)

#   Seperate the names to get names and release years.
#   Note that here are some specific text operations for a very few movies, which are precise instead of using string operating functions.
for name in mvname_list:
    #   for some names followed by a ` ` 
    if name[-1] == ' ':
        mvname_list[mvname_list.index(name)] = mvname_list[mvname_list.index(name)][:-1]
        name = name[0:-1]
    #   for some names containing no release year bracketed.
    if name[-1] != ')':
        releaseyear_list.append('')
    #   Seperate the name and release year and store them in different lists.
    else:
        #   for a name containing a peculiar form of release year.
        if mvname_list.index(name) != 9518:
            releaseyear_list.append(name[-5:-1])
            mvname_list[mvname_list.index(name)] = mvname_list[mvname_list.index(name)][:-7]
        else:
            releaseyear_list.append(name[-10:-6])
            mvname_list[mvname_list.index(name)] = mvname_list[mvname_list.index(name)][:-12]

#   Do pre-writing for titles.
#   Note that here isn't writing the lists finalized above.
with open(export_csv, 'a', encoding = 'utf-8', newline = '') as f:
    writer = csv.writer(f)
    #   The former five terms below have been collected so far while the later eleven haven't.
    #   The term `Poster URL` is for the list of links where the poster images are.
    writer.writerow(['movieId', 'imdbId', 'tmdbId', 'Name', 'Release Year', 'Genre', 'Duration', 'Release Date', 'Origin Country', 'Language', 'Filming Location', 'Director(s)', 'Writer(s)', 'Star(s)', 'Introduction', 'Poster URL'])


#     phase 3     Do crawling while writing info crawled into the export .csv file, logs into the log .txt files and posters into the poster folder.

#   Construct the poster folder in case it doesn't exist.
if not os.path.exists(poster_folder):
    os.makedirs(poster_folder)

#   Configure the arguments of log file, mainly for the time and variety of messages.
logging.basicConfig(filename = log_txt, filemode = 'a+', format = '%(asctime)s %(name)s: %(levelname)s: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S', level = logging.INFO)

#   Circularly crawl by movie ID.
for mvid in mvid_list:
    #   a debugging port, which must be commented in formal crawling
    # mvid = str(debugid)
    #   Pre-sleep aimed at the possible anti-crawling mechanism of the webpage, which could be omitted here since the targeted pages seem not to have such mechanism.
    time.sleep(1)
    #   Note the index, which is widely used below.
    index = mvid_list.index(mvid)
    #   Get the imdb ID from list by `index`.
    imdbid = imdbid_list[index]
    #   Form the usable movie link string by `tt` and the replenished senven-digit imdb ID by `0`(s).
    movie_url = url + 'tt' + '0' * (7 - len(imdbid)) + imdbid
    #   The image link is by the structure below for crawl the poster since there is no poster image in the main page of movies.
    #   Note that this is when the code file is construted and might be changed a lot in the aftertime.
    image_url = movie_url + image_url_suffix

    #   a log message for starting crawling the movie and a printing
    logging.info(f'Enter movie {mvid}. {mvname_list[index]} URL: {movie_url}')
    print(f'Enter movie {mvid}. {mvname_list[index]} URL: {movie_url}')
    #   Circulate for times.
    for launch in range(launch_time):
        try:
            #   More arguments can be configured here.
            movie_response = requests.get(url = movie_url, headers = headers, timeout = 6)
            #   Get the correct status-code, log, print and directly step forward.
            if movie_response.status_code == 200:
                # print(movie_response.content)
                logging.info(f'Succeed in movie {mvid}. {mvname_list[index]} {movie_url}')
                print(f'Succeed in movie {mvid}. {mvname_list[index]} {movie_url}')
                break
            #   In case of the status-code of the link error (often with `404`), which might not turn out to disappear for a short time, log, print and give up connecting to the movie main page directly and go on to the next step.
            else:
                logging.error(f'Fail with status-code error: {movie_response.status_code} in movie {mvid}. {mvname_list[index]} {movie_url}')
                print(f'Fail with status-code error: {movie_response.status_code} in movie {mvid}. {mvname_list[index]} {movie_url}')
                movie_response = None
                break
        except requests.RequestException:
            #   In case this launch time doesn't handle the request well, log, print and try again.
            if launch != launch_time - 1:
                logging.error(f'Fail for {launch + 1} time(s) in movie {mvid}. {mvname_list[index]} {movie_url} Trying again...')
                print(f'Fail for {launch + 1} time(s) in movie {mvid}. {mvname_list[index]} {movie_url} Trying again...')
            #   In case several times don't handle the request well, give up trying, log and print.
            else:
                logging.error(f'Fail with time-out error in movie {mvid}. {mvname_list[index]} {movie_url}')
                print(f'Fail with time-out error in movie {mvid}. {mvname_list[index]} {movie_url}')
                movie_response = None

    #   In case the movie main page fail, skip to the next movie.
    if movie_response == None:
        #   Write empty for the later eleven lists.
        with open(export_csv, 'a', encoding = 'utf-8', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([mvid, imdbid_list[index], tmdbid_list[index], mvname_list[index], releaseyear_list[index], '' * 11])
        #   Note the movie ID in the derived list.
        with open(derived_list_txt, 'a') as f:
            f.write(mvid + '\n')
        #   Note the movie ID in the failed list.
        with open(failed_list_txt, 'a') as f:
            f.write(mvid + '. ' + mvname_list[index] + ': Link broken.')
        continue

    #   a log message for starting crawling the poster and a printing
    logging.info(f'Enter image-page {mvid}. {mvname_list[index]} URL: {image_url}')
    print(f'Enter image-page {mvid}. {mvname_list[index]} URL: {image_url}')
    #   Circulate for times.
    for launch in range(launch_time):
        try:
            #   More arguments can be configured here.
            image_response = requests.get(url = image_url, headers = headers, timeout = 6)
            #   Get the correct status-code, log, print and directly step forward.
            if image_response.status_code == 200:
                logging.info(f'Succeed in image-page {mvid}. {mvname_list[index]} {image_url}')
                print(f'Succeed in image-page {mvid}. {mvname_list[index]} {image_url}')
                break
            #   In case of the status-code of the link error (often with `404`), which might not turn out to disappear for a short time, log, print and give up connecting to the image page directly and go on to the next step.
            else:
                logging.error(f'Fail with status-code error: {image_response.status_code} in image-page {mvid}. {mvname_list[index]} {image_url}')
                print(f'Fail with status-code error: {image_response.status_code} in image-page {mvid}. {mvname_list[index]} {image_url}')
                image_response = None
                break
        except requests.RequestException:
            #   In case this launch time doesn't handle the request well, log, print and try again.
            if launch != launch_time - 1:
                logging.error(f'Fail for {launch + 1} time(s) in image-page {mvid}. {mvname_list[index]} {image_url} Trying again...')
                print(f'Fail for {launch + 1} time(s) in image-page {mvid}. {mvname_list[index]} {image_url} Trying again...')
            #   In case several times don't handle the request well, give up trying, log and print.
            else:
                logging.error(f'Fail with time-out error in image-page {mvid}. {mvname_list[index]} {image_url}')
                print(f'Fail with time-out error in image-page {mvid}. {mvname_list[index]} {image_url}')
                image_response = None

    #   Derive the response content from the image page.
    image_soup = BeautifulSoup(image_response.content, 'lxml')

    #   Crawl the poster link to get the poster image.
    poster_url = ''
    try:
        #   Search `image_soup` for the class wherein the poster link might be.
        poster_url = image_soup.find(class_ = 'subpage_title_block').a.img['src']
        #   a log message for entering the poster image page and a printing
        logging.info(f'Enter poster URL: {poster_url}')
        print(f'Enter poster URL: {poster_url}')
        #   Circulate for times.
        for launch in range(launch_time):
            try:
                #   More arguments can be configured here.
                poster_response = requests.get(url = poster_url, headers = headers, timeout = 6)
                #   Get the correct status-code, log, print and directly step forward.
                if poster_response.status_code == 200:
                    logging.info(f'Succeed in poster {mvid}. {mvname_list[index]} {poster_url}')
                    print(f'Succeed in poster {mvid}. {mvname_list[index]} {poster_url}')
                    break
                #   In case of the status-code of the link error (often with `404`), which might not turn out to disappear for a short time, log, print and give up the poster crawling directly.
                else:
                    logging.error(f'Fail with status-code error: {poster_response.status_code} in poster {mvid}. {mvname_list[index]} {poster_url}')
                    print(f'Fail with status-code error: {poster_response.status_code} in poster {mvid}. {mvname_list[index]} {poster_url}')
                    poster_response = None
                    break
            except requests.RequestException:
                #   In case this launch time doesn't handle the request well, log, print and try again.
                if launch != launch_time - 1:
                    logging.error(f'Fail for {launch + 1} time(s) in poster {mvid}. {mvname_list[index]} {poster_url} Trying again...')
                    print(f'Fail for {launch + 1} time(s) in poster {mvid}. {mvname_list[index]} {poster_url} Trying again...')
                #   In case several times don't handle the request well, give up trying, log and print.
                else:
                    logging.error(f'Fail with time-out error in poster {mvid}. {mvname_list[index]} {poster_url}')
                    print(f'Fail with time-out error in poster {mvid}. {mvname_list[index]} {poster_url}')
                    poster_response = None
        #   Store the poster image crawled as .jpg file named by `mvid` and `imdbid`.
        with open(poster_folder + mvid + '_' + str(int(imdbid)) + '.jpg', 'wb') as f:
            f.write(poster_response.content)
    #   In case the search fails, write in the failed list file to show the poster is not found, log and print.
    except AttributeError as _:
        with open(failed_list_txt, 'a') as f:
            f.write(mvid + '. ' + mvname_list[index] + ': Poster not found.\n')
        logging.error(f'Not found poster of {mvid}. {mvname_list[index]}.')
        print(f'Not found poster of {mvid}. {mvname_list[index]}.')
    except TypeError as _:
        with open(failed_list_txt, 'a') as f:
            f.write(mvid + '. ' + mvname_list[index] + ': Poster not found.\n')
        logging.error(f'Not found poster of {mvid}. {mvname_list[index]}.')
        print(f'Not found poster of {mvid}. {mvname_list[index]}.')

    #   Derive the response content from the movie page.
    soup = BeautifulSoup(movie_response.content, 'lxml')

    #   Crawl the genre of the movie with normalization in text.
    #   a log message for beginning crawling the genre information and a printing
    logging.info(f'Begin to crawl the genre information of {mvid}. {mvname_list[index]}.')
    print(f'Begin to crawl the genre information of {mvid}. {mvname_list[index]}.')
    try:
        #   Search `soup` for the class wherein the genre information might be and seperate genres by `|`.
        genre = unicodedata.normalize('NFKC', soup.select_one('div[data-testid="genres"]').get_text(separator = '|').strip())
        #   Log the crawling result and print.
        if genre != '':
            logging.info(f'Succeed in deriving the genre information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the genre information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the genre information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the genre information of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        genre = ''
        logging.error(f'Fail deriving the genre information of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the genre information of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        genre = ''
        logging.error(f'Fail deriving the genre information of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the genre information of {mvid}. {mvname_list[index]} with type-error.')

    #   Crawl the duration of the movie with normalization in text.
    #   a log message for beginning crawling the duration and a printing
    logging.info(f'Begin to crawl the duration of {mvid}. {mvname_list[index]}.')
    print(f'Begin to crawl the duration of {mvid}. {mvname_list[index]}.')
    try:
        #   Search `soup` for the class wherein the duration information might be.
        duration = unicodedata.normalize('NFKC', soup.select_one('ul[data-testid="hero-title-block__metadata"]').find_all('li')[-1].get_text().strip())
        #   Prevent deriving a peculiar result with a wrong format.
        #   Log the crawling result and print.
        if duration != '':
            if duration[-1] != 'h' and duration[-1] != 'm':
                duration = ''
            logging.info(f'Succeed in deriving the duration of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the duration of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the duration of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the duration of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        duration = ''
        logging.error(f'Fail deriving the duration of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the duration of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        duration = ''
        logging.error(f'Fail deriving the duration of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the duration of {mvid}. {mvname_list[index]} with type-error.')

    #   Crawl the release date of the movie with normalization in text.
    #   a log message for beginning crawling the release date and a printing
    logging.info(f'Begin to crawl the release date of {mvid}. {mvname_list[index]}')
    print(f'Begin to crawl the release date of {mvid}. {mvname_list[index]}')
    try:
        #   Search `soup` for the class wherein the release date information might be.
        release_date = unicodedata.normalize('NFKC', soup.select_one('li[data-testid="title-details-releasedate"]').div.ul.li.a.get_text().strip().strip('"'))
        #   Log the crawling result and print.
        if release_date != '':
            #   Omit the release country bracketed.
            if release_date[-1] == ')':
                for sign in reversed(range(len(release_date))):
                    if release_date[sign] == '(':
                        release_date = release_date[:(sign - 1)]
                        break
            #   Prevent deriving a peculiar result with a wrong format.
            if release_date[-4] != '1' and release_date[-4] != '2':
                release_date = ''
            if release_date[-3] != '9' and release_date[-3] != '0':
                release_date = ''
            logging.info(f'Succeed in deriving the release date of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the release date of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the release date of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the release date of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        release_date = ''
        logging.error(f'Fail deriving the release date of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the release date of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        release_date = ''
        logging.error(f'Fail deriving the release date of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the release date of {mvid}. {mvname_list[index]} with type-error.')

    #   Crawl the origin country of the movie with normalization in text.
    #   a log message for beginning crawling the origin country information and a printing
    logging.info(f'Begin to crawl the origin country information of {mvid}. {mvname_list[index]}')
    print(f'Begin to crawl the origin country information of {mvid}. {mvname_list[index]}')
    try:
        #   Search `soup` for the class wherein the country information might be.
        origin = unicodedata.normalize('NFKC', soup.select_one('li[data-testid="title-details-origin"]').div.ul.li.a.get_text().strip().strip('"'))
        #   Log the crawling result and print.
        if origin != '':
            logging.info(f'Succeed in deriving the origin country information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the origin country information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the origin country information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the origin country information of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        origin = ''
        logging.error(f'Fail deriving the origin country information of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the origin country information of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        origin = ''
        logging.error(f'Fail deriving the origin country information of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the origin country information of {mvid}. {mvname_list[index]} with type-error.')

    #   Crawl the language of the movie with normalization in text.
    #   a log message for beginning crawling the language information and a printing
    logging.info(f'Begin to crawl the language information of {mvid}. {mvname_list[index]}')
    print(f'Begin to crawl the language information of {mvid}. {mvname_list[index]}')
    try:
        #   Search `soup` for the class wherein the language information might be.
        language = unicodedata.normalize('NFKC', soup.select_one('li[data-testid="title-details-languages"]').div.ul.li.a.get_text().strip().strip('"'))
        #   Log the crawling result and print.
        if language != '':
            logging.info(f'Succeed in deriving the language information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the language information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the language information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the language information of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        language = ''
        logging.error(f'Fail deriving the language information of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the language information of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        language = ''
        logging.error(f'Fail deriving the language information of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the language information of {mvid}. {mvname_list[index]} with type-error.')

    #   Crawl the filming location of the movie with normalization in text.
    #   a log message for beginning crawling the filming location information and a printing
    logging.info(f'Begin to crawl the filming location information of {mvid}. {mvname_list[index]}')
    print(f'Begin to crawl the filming location information of {mvid}. {mvname_list[index]}')
    try:
        #   Search `soup` for the class wherein the filming location information might be.
        filming_location = unicodedata.normalize('NFKC', soup.select_one('li[data-testid="title-details-filminglocations"]').div.ul.li.a.get_text().strip().strip('"'))
        #   Log the crawling result and print.
        if filming_location != '':
            logging.info(f'Succeed in deriving the filming location information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the filming location information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the filming location information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the filming location information of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        filming_location = ''
        logging.error(f'Fail deriving the filming location information of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the filming location information of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        filming_location = ''
        logging.error(f'Fail deriving the filming location information of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the filming location information of {mvid}. {mvname_list[index]} with type-error.')

    #   Crawl the cast information of the movie with normalization in text.
    #   a log message for beginning crawling the cast information and a printing
    logging.info(f'Begin to crawl the cast information of {mvid}. {mvname_list[index]}')
    print(f'Begin to crawl the cast information of {mvid}. {mvname_list[index]}')
    try:
        #   Search `soup` for the class wherein the cast information might be and store in the cast list.
        cast_list = soup.find_all('li', attrs = {'data-testid': "title-pc-principal-credit"})
        if cast_list != None:
            director = []
            scriptwriter = []
            star = []
            #   Collect each kind of cast by the order.
            director_list = cast_list[0].div.ul.find_all('li')
            scriptwriter_list = cast_list[1].div.ul.find_all('li')
            star_list = cast_list[2].div.ul.find_all('li')
            #   Circulate for each list to append cast name(s).
            for role in director_list:
                director.append(unicodedata.normalize('NFKC', role.a.get_text().strip()))
            for role in scriptwriter_list:
                scriptwriter.append(unicodedata.normalize('NFKC', role.a.get_text().strip()))
            for role in star_list:
                star.append(unicodedata.normalize('NFKC', role.a.get_text().strip()))
            #   Seperate each by `|`.
            director = '|'.join(director)
            scriptwriter = '|'.join(scriptwriter)
            star = '|'.join(star)
        #   Log the crawling result and print.
        if director != '':
            logging.info(f'Succeed in deriving the director information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the director information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the director information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the director information of {mvid}. {mvname_list[index]}.')
        if scriptwriter != '':
            logging.info(f'Succeed in deriving the scriptwriter information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the scriptwriter information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the scriptwriter information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the scriptwriter information of {mvid}. {mvname_list[index]}.')
        if star != '':
            logging.info(f'Succeed in deriving the star information of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the star information of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the star information of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the star information of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        director = ''
        scriptwriter = ''
        star = ''
        logging.error(f'Fail deriving the cast information of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the cast information of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        director = ''
        scriptwriter = ''
        star = ''
        logging.error(f'Fail deriving the cast information of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the cast information of {mvid}. {mvname_list[index]} with type-error.')
    except IndexError as _:
        director = ''
        scriptwriter = ''
        star = ''
        logging.error(f'Fail deriving the cast information of {mvid}. {mvname_list[index]} with index-error.')
        print(f'Fail deriving the cast information of {mvid}. {mvname_list[index]} with index-error.')

    #   Crawl the introduction of the movie with normalization in text.
    #   a log message for beginning crawling the introduction and a printing
    logging.info(f'Begin to crawl the introduction of {mvid}. {mvname_list[index]}')
    print(f'Begin to crawl the introduction of {mvid}. {mvname_list[index]}')
    try:
        #   Search `soup` for the class wherein the genre and release inroduction might be.
        intro = unicodedata.normalize('NFKC', soup.select_one('div[data-testid="storyline-plot-summary"]').div.div.get_text().strip().strip('"'))
        #   Log the crawling result and print.
        if intro != '':
            #   Omit the possible quote source.
            if intro[-1] != '.':
                flag = False
                #   Deal with a possible e-mail bracketed by `<` and `>`.
                if intro[-1] == '>':
                    flag = True
                for sign in reversed(range(len(intro))):
                    if intro[sign] != '<' and flag:
                        continue
                    if intro[sign] == '<':
                        flag = False
                        continue
                    if intro[sign] == '.' and not flag:
                        intro = intro[:(sign + 1)]
                        break
            logging.info(f'Succeed in deriving the introduction of {mvid}. {mvname_list[index]}.')
            print(f'Succeed in deriving the introduction of {mvid}. {mvname_list[index]}.')
        else:
            logging.info(f'Fail deriving the introduction of {mvid}. {mvname_list[index]}.')
            print(f'Fail deriving the introduction of {mvid}. {mvname_list[index]}.')
    #   in case the search fails with a log message and a printing
    except AttributeError as _:
        intro = ''
        logging.error(f'Fail deriving the introduction of {mvid}. {mvname_list[index]} with attribute-error.')
        print(f'Fail deriving the introduction of {mvid}. {mvname_list[index]} with attribute-error.')
    except TypeError as _:
        intro = ''
        logging.error(f'Fail deriving the introduction of {mvid}. {mvname_list[index]} with type-error.')
        print(f'Fail deriving the introduction of {mvid}. {mvname_list[index]} with type-error.')

    #   Write all the information into the row corresponding to the movie.
    try:
        with open(export_csv, 'a', encoding = 'utf-8', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([mvid, imdbid_list[index], tmdbid_list[index], mvname_list[index], releaseyear_list[index], genre, duration, release_date, origin, language, filming_location, director, scriptwriter, star, intro, poster_url])
    #   In case of the encode-error, write empty, log and print.
    except UnicodeEncodeError as _:
        with open(export_csv, 'a', encoding = 'utf-8', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([mvid, imdbid_list[index], tmdbid_list[index], mvname_list[index], releaseyear_list[index], '' * 11])
        logging.error(f'Fail writing the information of {mvid}. {mvname_list[index]} with unicode-encode-error.')
        print(f'Fail writing the information of {mvid}. {mvname_list[index]} with unicode-encode-error.')

    #   Note the movie ID in the derived list.
    with open(derived_list_txt, 'a') as f:
        f.write(mvid + '\n')

    #   a log message for finishing crawling the movie and a printing
    logging.info(f'Succeed movie {mvid}. {mvname_list[index]}.')
    print(f'Succeed movie {mvid}. {mvname_list[index]}.')
    logging.info(f'{len(mvid_list) - mvid_list.index(mvid) - 1} movie(s) left.\n')
    print(f'{len(mvid_list) - mvid_list.index(mvid) - 1} movie(s) left.\n')

#   Set a final log message and a printing for finishing crawling.
logging.info(f'Crawl done!')
print(f'Crawl done!')


#     phase 4     Collect all genres from movie information crawled and store the genre list in a .txt file and a .csv file.

#   Take the genre list.
with open(export_csv, encoding = 'utf-8') as f:
    reader = csv.reader(f)
    #   Skip the title.
    _ = reader.__next__()
    for row in reader:
        if row[5]:
            genre_set.update(row[5].split('|'))
#   Sort the genre list in lexicographical order.
genre_list = sorted(genre_set)

#   Write the .txt file by line.
with open(genre_list_txt, 'w') as f:
    for genre in genre_set:
        f.write(genre + '\n')

#   Write the .csv file by row.
with open(genre_list_csv, 'w', newline = '') as f:
    writer = csv.writer(f)
    for row in zip(genre_list):
        writer.writerow(row)
