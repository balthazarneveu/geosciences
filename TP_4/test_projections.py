import torch
import pytest
from plane_cylinder_projections import normal_vector_to_angles, angles_to_normal_vector
import math
import torch


def test_normal_vector_to_angles():
    # Test case 1: Normal vector pointing in the x-axis direction
    vec_normal = torch.tensor([1.0, 0.0, 0.0])
    expected_angles = torch.tensor([torch.pi/2, 0.0])
    assert torch.allclose(normal_vector_to_angles(vec_normal), expected_angles)

    # Test case 2: Normal vector pointing in the y-axis direction
    vec_normal = torch.tensor([0.0, 1.0, 0.0])
    expected_angles = torch.tensor([torch.pi/2, torch.pi/2])
    assert torch.allclose(normal_vector_to_angles(vec_normal), expected_angles)

    # Test case 3: Normal vector pointing in the z-axis direction
    vec_normal = torch.tensor([0.0, 0.0, 1.0])
    expected_angles = torch.tensor([0.0, 0.0])
    assert torch.allclose(normal_vector_to_angles(vec_normal), expected_angles)


def test_angles_to_normal_vector():
    atol = 1e-7
    # Test case 1: Angles representing a normal vector pointing in the x-axis direction
    angles = torch.tensor([0.0, torch.pi/2])
    expected_normal_vector = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(angles_to_normal_vector(angles), expected_normal_vector, atol=atol)

    # Test case 2: Angles representing a normal vector pointing in the y-axis direction
    angles = torch.tensor([torch.pi/2, torch.pi/2])
    expected_normal_vector = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(angles_to_normal_vector(angles), expected_normal_vector, atol=atol)

    # Test case 3: Angles representing a normal vector pointing in the z-axis direction
    angles = torch.tensor([0.0, 0.0])
    expected_normal_vector = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(angles_to_normal_vector(angles), expected_normal_vector, atol=atol)

    # Additional test case
    angles = torch.tensor([torch.pi/4, torch.pi/4])
    expected_normal_vector = torch.tensor([0.5, 0.5, 0.7071])
    assert torch.allclose(angles_to_normal_vector(angles), expected_normal_vector, atol=atol)


def test_cycle_of_functions():
    atol = 1e-7
    # Test case 1: Angles -> Vector -> Angles
    angles = torch.tensor([torch.pi/4, torch.pi/4])
    assert torch.allclose(normal_vector_to_angles(angles_to_normal_vector(angles)), angles, atol=atol)

    # Test case 2: Angles -> Vector -> Angles
    angles = torch.tensor([torch.pi/2, 0.])
    assert torch.allclose(normal_vector_to_angles(angles_to_normal_vector(angles)), angles, atol=atol)

    # Test case 3: Angles -> Vector -> Angles
    angles = torch.tensor([torch.pi/2, torch.pi/2])
    assert torch.allclose(normal_vector_to_angles(angles_to_normal_vector(angles)), angles, atol=atol)

    # Test case 4: Angles -> Vector -> Angles
    angles = torch.tensor([0.0, 0.0])
    assert torch.allclose(normal_vector_to_angles(angles_to_normal_vector(angles)), angles, atol=atol)
