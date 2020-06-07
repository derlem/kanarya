from django.db import models
from django.contrib.auth.models import User
from PIL import Image

class Profile(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	image = models.ImageField(default='default.jpg', upload_to='profile_pics')
	last_seen_sentence_idx = models.IntegerField(default=1)
	correct_answer_count = models.IntegerField(default=0)

	#onam = models.BooleanField(default=False)
	#isOnamSubmitted = models.BooleanField(default=False)

	#last_seen_prof_idx = models.IntegerField(default=1)
	#prof_score = models.IntegerField(default=0)
	#is_prof_done = models.BooleanField(default=False)

	last_seen_warmup_idx = models.IntegerField(default=1)
	is_warmup_done = models.BooleanField(default=False)

	tos = models.BooleanField(default=False) # terms of service (onam formu)

	def __str__(self):
		return f'{self.user.username} Profile'

	
	def save(self, *args, **kwargs):
		super().save(*args, **kwargs)

		img = Image.open(self.image.path)

		if img.height > 300 or img.width > 300:
			output_size = (300,300)
			img.thumbnail(output_size)
			img.save(self.image.path)


	