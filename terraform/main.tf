
provider "aws" {
     region = var.my_region

     required_providers {
        aws={
            source  = "hashicorp/aws"
            version = "~> 4.0"
        }
     }  
}

resource "aws_vpc" "main_vpc"{
    cidr_block            = var.vpc_cidr
    enable_dns_support    = true
    enable_dns_hostnames  = true

  tags = {
    Name = "main-vpc"
  }
}

resource "aws_internet_gateway" "igw" {
    vpc_id = aws_vpc.main_vpc.id

tags = {

    Name = "main-igw"
}
}

resource "aws_subnet" "public_subnet" {
    vpc_id                   = aws_vpc.main_vpc.id 
    cidr_block               = var.public_subnet_cidr
    availability_zone        = var.aws_availability_zone
    map_public_ip_on_launch  = true
    
 }

 resource "aws_route_table" "public_rt" {
    vpc_id = aws_vpc.main_vpc.id

    route {

        cidr_block = "0.0.0.0/0"
        gateway_id = aws_internet_gateway.igw.id
    }

   tags {
            Name = "public-rt"
    }
}

resource "aws_route_table_association" "public_rt_assoc" {
    subnet_id       = aws_subnet.public_subnet.id
    route_table_id  = aws_route_table.public_rt.id
}

resource "aws_security_group" "gpu_instance_sg" {
    name = "gpu-instance-sg"
    vpc_id = aws_vpc.main_vpc.id

    ingress {
        description = "Allow SSH"
        from_port   =  22
        to_port     =  22
        protocol    =  "tcp"
        cidr_block  =  ["0.0.0.0/0"]
    }

    ingress {
        description = "Allow container app port"
        from_port   =  var.app_port
        to_port     =  var.app_port
        protocol    =  "tcp"
        cidr_block  =  ["0.0.0.0/0"]
    }

    egress {
        from_port   = 0
        to_port     = 0
        protocol    = "-1"
        cidr_block  = ["0.0.0.0/0"]
    }

    tags = {
        Name = "gpu_instance_sg"
    }
}

