import chi
import chi.lease
import chi.server
import chi.network
import chi.blockstorage
from chi.ssh_keys import add_project_ssh_key

# Config
PROJECT_ID = "CHI-251409"
KEY_NAME = "kathan-key"
PUBLIC_KEY = """ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDVXSSnrG/enTZMLlaCe66jbhQjHhPVzSL/qYWBNFy/rnZwtD6Ipkm+MKo73vvDaoKbIeSUf9G+isINBnKd0iTGJDjLhMN5+iFzRYgkheSHJt0xFxDvub8jtRRx04ziLemtWk85OEIYMSEn4KOEeaETdexCX/I6YlwPTbkOQKUjXCHE7ay20KDcjCQF9L53ElOO3xYgQq2prY1kGas4/6jqTYI4OmmH3PnT7yWFptimSowCq7zUYJJJHQVH3oDoiBEsPKkR9fJpLh8vRFkxtsyTcvHmkFpfMEd2gzY5a4v+iTPKLjSKJD+Xt4C+LerdSOEMSlp8TzlLT04uyuU/5Tknu5AmdVUt1e9vdI/O8Mehlk8VkqDUcBtuVNQb6C8jBUoN+LhD9NKmP4R7hCGmOO5jc5t2mFPe/GCveCPMzeuRJNwbbISMMGjtR7hjGrRKGztNvPB4J0AUX4pB95TCTVTQD7hb3lfI8kYjMLwJVhj+z1Va6ztFKrnQhQiUQKB5IV0= katha@KathanDPredator"""

IMAGE_NAME = "CC-Ubuntu20.04"
FLAVOR_NAME = "m1.medium"
NETWORK_NAME = "sharednet1-project20"
INSTANCE_NAMES = ["k8s-master-project20", "k8s-worker1-project20", "k8s-worker2-project20"]

# 1. Set project context
chi.set("project_name", PROJECT_ID)

# 2. Upload SSH key (skip if already uploaded manually)
add_project_ssh_key(key_name=KEY_NAME, public_key=PUBLIC_KEY)

# 3. Get image/flavor
image = chi.get_image(name=IMAGE_NAME)
flavor = chi.get_flavor(name=FLAVOR_NAME)

# 4. Create or get network
network = chi.network.create_or_get_network(name=NETWORK_NAME)

# 5. Launch instances
for name in INSTANCE_NAMES:
    chi.server.launch(
        name=name,
        image=image,
        flavor=flavor,
        networks=[network],
        key_name=KEY_NAME
    )

print(f"✅ Launched VMs: {INSTANCE_NAMES} on network {NETWORK_NAME}")
