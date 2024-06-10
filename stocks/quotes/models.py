from django.db import models

# Create your models here.
class Stock(models.Model):
    ticker = models.CharField(max_length=10)
    # stock_type = models.CharField(max_length=20) # pharma tech motor
    def __str__(self) -> str:
        return self.ticker