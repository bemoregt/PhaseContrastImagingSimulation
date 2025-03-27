import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import fftpack
from skimage import data
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_phase_object(size=256, num_objects=5, max_phase=2*np.pi):
    """투명한 위상 물체를 시뮬레이션합니다."""
    phase = np.zeros((size, size))
    
    # 무작위 위치에 여러 개의 가우시안 모양의 위상 물체 생성
    for _ in range(num_objects):
        x0, y0 = np.random.randint(0, size, 2)
        sigma = np.random.uniform(5, 20)
        amplitude = np.random.uniform(0.2, 1.0) * max_phase
        
        y, x = np.ogrid[-y0:size-y0, -x0:size-x0]
        r2 = x*x + y*y
        gaussian = amplitude * np.exp(-r2 / (2 * sigma**2))
        phase += gaussian
    
    return phase

def propagate_beam(complex_amplitude, distance, wavelength=0.5e-6, pixel_size=1e-6):
    """파동 전파 시뮬레이션 (각 스펙트럼 방법 사용)"""
    rows, cols = complex_amplitude.shape
    
    # 주파수 그리드 생성
    kx = 2 * np.pi * fftpack.fftfreq(cols, d=pixel_size)
    ky = 2 * np.pi * fftpack.fftfreq(rows, d=pixel_size)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength
    
    # 각 스펙트럼 방법 전파자 (angular spectrum propagator)
    kz_grid = np.sqrt(k**2 - kx_grid**2 - ky_grid**2 + 0j)
    propagator = np.exp(1j * kz_grid * distance)
    
    # 전파 파면의 불필요한 발산을 방지하기 위한 필터
    mask = (kx_grid**2 + ky_grid**2) <= k**2
    propagator = propagator * mask
    
    # 푸리에 변환, 전파, 역변환
    ft = fftpack.fft2(complex_amplitude)
    ft_propagated = ft * propagator
    return fftpack.ifft2(ft_propagated)

def generate_defocus_images(phase_object, defocus_distances, wavelength=0.5e-6, pixel_size=1e-6):
    """여러 초점 거리에서의 이미지를 생성합니다."""
    # 입사 파동을 평면파로 가정 (진폭=1, 위상=0)
    amplitude = np.ones_like(phase_object)
    complex_amplitude = amplitude * np.exp(1j * phase_object)
    
    defocus_images = []
    for distance in defocus_distances:
        # 특정 거리만큼 전파
        propagated = propagate_beam(complex_amplitude, distance, wavelength, pixel_size)
        
        # 강도 이미지 계산 (복소 진폭의 절대값 제곱)
        intensity = np.abs(propagated)**2
        defocus_images.append(intensity)
    
    return np.array(defocus_images)

def solve_tie(defocus_images, delta_z, wavelength=0.5e-6):
    """Transport of Intensity Equation을 사용하여 위상을 복원합니다."""
    # 중앙 평면 이미지 (초점면)
    I0 = defocus_images[1]
    
    # 강도의 z-축 미분 계산 (중앙 차분법)
    dI_dz = (defocus_images[2] - defocus_images[0]) / (2 * delta_z)
    
    # TIE 방정식 해결 (푸리에 방법 사용)
    rows, cols = I0.shape
    kx = 2 * np.pi * fftpack.fftfreq(cols)
    ky = 2 * np.pi * fftpack.fftfreq(rows)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    # 라플라시안의 역연산자 (0 주파수에서의 특이점 처리)
    k_squared = kx_grid**2 + ky_grid**2
    k_squared[0, 0] = 1  # 0으로 나누기 방지
    
    # TIE 방정식 해결
    ft_dI_dz = fftpack.fft2(dI_dz)
    ft_phase = -wavelength / (2 * np.pi) * ft_dI_dz / (k_squared * I0.mean())
    ft_phase[0, 0] = 0  # DC 성분 제거
    
    # 위상 복원
    phase = np.real(fftpack.ifft2(ft_phase))
    
    return phase

def simulate_phase_contrast(phase_image, amplitude_image=None, phase_ring_radius=10, phase_shift=np.pi/2, contrast_enhance=5.0):
    """위상 대비 현미경을 시뮬레이션합니다."""
    # 진폭 이미지가 없으면 균일한 진폭 가정
    if amplitude_image is None:
        amplitude_image = np.ones_like(phase_image)
    
    # 복소 이미지 생성 (진폭과 위상 결합)
    complex_image = amplitude_image * np.exp(1j * phase_image)
    
    # 푸리에 변환
    fourier = fftpack.fft2(complex_image)
    fourier_shifted = fftpack.fftshift(fourier)
    
    # 위상판 효과 생성
    rows, cols = phase_image.shape
    center_row, center_col = rows//2, cols//2
    
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    dist_from_center = np.sqrt(x*x + y*y)
    
    # 위상판 마스크 생성 (중앙에 위상 지연 및 진폭 감쇠 모두 적용 - 실제 위상 대비 현미경과 유사하게)
    phase_plate = np.ones_like(fourier_shifted, dtype=complex)
    
    # 위상 지연 효과 강화 (음성 위상 대비 효과)
    phase_plate[dist_from_center <= phase_ring_radius] = np.exp(1j * phase_shift) * 0.5  # 진폭 감쇠 추가
    
    # 위상판 효과 적용
    filtered_fourier = fourier_shifted * phase_plate
    
    # 역 푸리에 변환
    filtered_fourier_shifted = fftpack.ifftshift(filtered_fourier)
    inverse_fourier = fftpack.ifft2(filtered_fourier_shifted)
    
    # 강도 이미지 계산
    intensity_image = np.abs(inverse_fourier)**2
    
    # 대비 향상
    intensity_mean = intensity_image.mean()
    intensity_enhanced = intensity_mean + (intensity_image - intensity_mean) * contrast_enhance
    
    # 정규화 (값이 0~1 범위에 들어가도록)
    intensity_min = intensity_enhanced.min()
    intensity_max = intensity_enhanced.max()
    intensity_normalized = (intensity_enhanced - intensity_min) / (intensity_max - intensity_min)
    
    return intensity_normalized

# 메인 시뮬레이션 코드
if __name__ == "__main__":
    # 파라미터 설정
    size = 256  # 이미지 크기
    wavelength = 0.5e-6  # 파장 (0.5 μm, 초록색 빛)
    pixel_size = 1e-6  # 픽셀 크기 (1 μm)
    delta_z = 20e-6  # 초점 거리 간격 (20 μm) - 차이를 더 명확하게
    
    # 시뮬레이션 결과 재현성을 위한 랜덤 시드 설정
    np.random.seed(42)
    
    # 세 개의 초점 거리 설정 (-delta_z, 0, +delta_z)
    defocus_distances = np.array([-delta_z, 0, delta_z])
    
    # 위상 물체 생성 - 더 강한 위상 효과를 위해 조정
    original_phase = generate_phase_object(size=size, num_objects=5, max_phase=3*np.pi)
    
    # 여러 초점 거리에서의 이미지 생성
    defocus_images = generate_defocus_images(
        original_phase, 
        defocus_distances, 
        wavelength, 
        pixel_size
    )
    
    # 초점면 이미지 - 완벽한 투명 물체 (진폭 변화 없음)
    focus_plane = np.ones((size, size), dtype=np.float64)
    
    # TIE로 위상 복원
    reconstructed_phase = solve_tie(defocus_images, delta_z, wavelength)
    
    # 위상 대비 현미경 시뮬레이션 - 대비 증가
    phase_contrast_image = simulate_phase_contrast(
        reconstructed_phase, 
        amplitude_image=None,
        phase_ring_radius=5, 
        phase_shift=np.pi/2,
        contrast_enhance=10.0  # 높은 대비 값
    )
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original_phase, cmap='viridis')
    axes[0, 0].set_title('original phase object')
    
    axes[0, 1].imshow(defocus_images[0], cmap='gray')
    axes[0, 1].set_title('focus front image')
    
    axes[0, 2].imshow(focus_plane, cmap='gray')
    axes[0, 2].set_title('focus plane image')
    
    axes[1, 0].imshow(defocus_images[2], cmap='gray')
    axes[1, 0].set_title('focus backward image')
    
    axes[1, 1].imshow(reconstructed_phase, cmap='viridis')
    axes[1, 1].set_title('reconstructed phase using TIE')
    
    axes[1, 2].imshow(phase_contrast_image, cmap='gray')
    axes[1, 2].set_title('phase contrast microscope simulation')
    
    plt.tight_layout()
    plt.show()
    
    # 위상 대비 이미지와 일반 이미지 비교
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(focus_plane, cmap='gray')
    plt.title('bright-field image')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(phase_contrast_image, cmap='gray')
    plt.title('phase contrast image')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()