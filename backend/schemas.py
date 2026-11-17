from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field

# --- Competitor Profile Schemas ---
class CompetitorBase(BaseModel):
    username: str
    nickname: Optional[str] = None
    profile_url: Optional[str] = None
    niche: Optional[str] = None
    notes: Optional[str] = None
    avatar_url: Optional[str] = None
    is_own_profile: Optional[bool] = False

class CompetitorCreate(CompetitorBase):
    pass

class CompetitorUpdate(BaseModel):
    nickname: Optional[str] = None
    niche: Optional[str] = None
    notes: Optional[str] = None
    avatar_url: Optional[str] = None

class CompetitorResponse(CompetitorBase):
    id: str
    created_at: datetime
    updated_at: datetime
    video_count: Optional[int] = 0

    class Config:
        from_attributes = True


# --- Product Cluster Schemas ---
class ProductClusterBase(BaseModel):
    name: str
    category: Optional[str] = None
    target_audience: Optional[str] = None
    description: Optional[str] = None

class ProductClusterCreate(ProductClusterBase):
    pass

class ProductClusterUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    target_audience: Optional[str] = None
    description: Optional[str] = None

class ProductClusterResponse(ProductClusterBase):
    id: str
    created_at: datetime
    video_count: Optional[int] = 0
    idea_count: Optional[int] = 0

    class Config:
        from_attributes = True


# --- Metric Snapshot Schemas ---
class MetricSnapshotBase(BaseModel):
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    extraction_source: Optional[str] = "manual"

class MetricSnapshotCreate(MetricSnapshotBase):
    pass

class MetricSnapshotResponse(MetricSnapshotBase):
    id: str
    video_id: str
    recorded_at: datetime

    class Config:
        from_attributes = True


# --- Video Analysis Schemas ---
class VideoAnalysisResponse(BaseModel):
    id: str
    video_id: str
    hook_initial: Optional[str] = None
    headline: Optional[str] = None
    pain_or_desire: Optional[str] = None
    sales_argument: Optional[str] = None
    video_structure: Optional[str] = None
    product_demo_style: Optional[str] = None
    cta_type: Optional[str] = None
    hook_strength_score: Optional[float] = None
    key_takeaways: Optional[str] = None
    raw_analysis_json: Optional[str] = None
    analyzed_at: datetime

    class Config:
        from_attributes = True

class AnalyzeVideoRequest(BaseModel):
    video_id: str
    transcript_override: Optional[str] = None
    custom_prompt_id: Optional[str] = None


# --- Video Asset Schemas ---
class VideoAssetCreate(BaseModel):
    tiktok_url: str
    competitor_id: Optional[str] = None
    product_cluster_id: Optional[str] = None
    # Optional manual overrides / metrics on registration
    title_caption: Optional[str] = None
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    published_at: Optional[datetime] = None
    raw_transcript: Optional[str] = None
    extraction_source: Optional[str] = "manual"

class VideoAssetUpdate(BaseModel):
    product_cluster_id: Optional[str] = None
    competitor_id: Optional[str] = None
    title_caption: Optional[str] = None
    raw_transcript: Optional[str] = None
    published_at: Optional[datetime] = None

class VideoAssetResponse(BaseModel):
    id: str
    competitor_id: Optional[str] = None
    product_cluster_id: Optional[str] = None
    tiktok_url: str
    tiktok_video_id: Optional[str] = None
    title_caption: Optional[str] = None
    author_name: Optional[str] = None
    author_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    published_at: Optional[datetime] = None
    raw_transcript: Optional[str] = None
    status: str
    created_at: datetime

    latest_snapshot: Optional[MetricSnapshotResponse] = None
    analysis: Optional[VideoAnalysisResponse] = None
    competitor_username: Optional[str] = None
    product_name: Optional[str] = None

    class Config:
        from_attributes = True


# --- Pattern Matching & Comparison Schemas ---
class VideoPerformanceItem(BaseModel):
    video_id: str
    tiktok_url: str
    title_caption: Optional[str] = None
    thumbnail_url: Optional[str] = None
    creator_username: Optional[str] = None
    is_own_video: Optional[bool] = False
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    engagement_rate: Optional[float] = None
    hook_initial: Optional[str] = None
    headline: Optional[str] = None
    video_structure: Optional[str] = None
    sales_argument: Optional[str] = None
    cta_type: Optional[str] = None
    hook_strength_score: Optional[float] = None

class PatternAnalysisResponse(BaseModel):
    product_cluster_id: str
    product_name: str
    total_videos_analyzed: int
    top_performing_videos: List[VideoPerformanceItem]
    common_winning_hooks: List[str]
    dominant_structures: List[str]
    key_selling_arguments: List[str]
    engagement_insights: str
    synthesis_report: str


# --- Video Idea Generation Schemas ---
class GenerateIdeasRequest(BaseModel):
    product_cluster_id: str
    count: Optional[int] = 3
    custom_instructions: Optional[str] = None

class VideoIdeaResponse(BaseModel):
    id: str
    product_cluster_id: str
    angle_title: str
    hook_suggestion: str
    proposed_structure: Optional[str] = None
    differentiation_angle: str
    script_outline: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Prompt Template Schemas ---
class PromptTemplateBase(BaseModel):
    name: str
    template_type: str
    system_prompt: Optional[str] = None
    template_content: str
    is_active: Optional[bool] = True

class PromptTemplateCreate(PromptTemplateBase):
    pass

class PromptTemplateUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    template_content: Optional[str] = None
    is_active: Optional[bool] = None

class PromptTemplateResponse(PromptTemplateBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
