from django.db import models


class Document(models.Model):
    file = models.FileField()
