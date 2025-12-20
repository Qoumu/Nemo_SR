# Create a non-root user
adduser myuser
usermod -aG audio myuser
usermod -aG sudo myuser

# Switch to that user
su - myuser

# Then run your Python scripts