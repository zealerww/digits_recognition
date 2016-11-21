# Handwrtten Digits Recognition

It is a web application of handwritten digits recognition based on Django. The website looks like:
![website](https://raw.githubusercontent.com/zealerww/digits_recognition/master/digits_recognition/static/website.png)

It's deployed on [Heroku](https://www.heroku.com/home)

[Demo Website](https://murmuring-ravine-30623.herokuapp.com/)


## Dependencies

* Python
* Django
* PIL
* Tensorflow
* whitenoise
* gunicorn

## Local

You can run the app locally

	$ python3 manage.py runserver 6000

And you can visit it on localhost:6000

## Deploy to Heroku

You can deploy the app to Heroku if you have a account

	$ heroku login
	$ heroku create
	$ git push heroku master



