# Generated by Django 3.0.5 on 2020-05-27 18:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('spellchecker', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='query',
            name='isHappy',
            field=models.BooleanField(blank=True, null=True),
        ),
    ]
