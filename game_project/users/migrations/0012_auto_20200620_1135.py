# Generated by Django 3.0.5 on 2020-06-20 11:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0011_auto_20200607_1922'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='last_seen_sentence_idx',
            field=models.IntegerField(default=0),
        ),
    ]